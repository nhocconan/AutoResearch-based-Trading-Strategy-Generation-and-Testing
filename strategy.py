#!/usr/bin/env python3
"""
Experiment #019: 4h Mean Reversion with 1d Trend Bias + Taker Volume Sentiment

Hypothesis: Trend-following strategies (KAMA, ADX, Donchian) have failed repeatedly 
on BTC/ETH in bear/range markets. Mean reversion with HTF trend filter should work better.

Key innovations:
1. Taker Buy Volume Ratio as sentiment extreme indicator (contrarian signal)
2. RSI(14) extremes for mean reversion entry (not trend confirmation)
3. Bollinger Band(20, 2.0) touch confirmation
4. 1d HMA(21) for trend bias (only trade WITH HTF trend)
5. Looser entry thresholds to ensure 30+ trades/year on 4h

Why this should work:
- Mean reversion excels in range/bear markets (2025 test period)
- Taker volume extremes predict short-term reversals (crowd is wrong at extremes)
- HTF trend filter prevents dangerous counter-trend trades
- 4h TF targets 20-50 trades/year (fee-efficient)

Timeframe: 4h (REQUIRED)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_meanrev_taker_rsi_bb_1d_trend_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_taker_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio (sentiment indicator)."""
    ratio = taker_buy_volume / np.where(volume == 0, 1e-10, volume)
    ratio = np.clip(ratio, 0.0, 1.0)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend bias
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    taker_ratio = calculate_taker_ratio(taker_buy_vol, volume)
    
    # Taker ratio rolling stats for extreme detection
    taker_ratio_s = pd.Series(taker_ratio)
    taker_ma = taker_ratio_s.rolling(window=20, min_periods=20).mean().values
    taker_std = taker_ratio_s.rolling(window=20, min_periods=20).std().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        if np.isnan(taker_ratio[i]) or np.isnan(taker_ma[i]):
            continue
        
        # === HTF TREND BIAS (1d) ===
        # Only trade WITH the daily trend
        htf_bullish = close[i] > hma_1d_21_aligned[i]
        htf_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H MEAN REVERSION SIGNALS ===
        # Price at Bollinger Band extreme
        at_bb_lower = close[i] <= bb_lower[i] * 1.002  # Within 0.2% of lower band
        at_bb_upper = close[i] >= bb_upper[i] * 0.998  # Within 0.2% of upper band
        
        # RSI extreme (oversold/overbought)
        rsi_oversold = rsi_14[i] < 38  # Looser than 30 for more trades
        rsi_overbought = rsi_14[i] > 62  # Looser than 70 for more trades
        
        # Taker volume sentiment extreme (contrarian)
        taker_extreme_low = taker_ratio[i] < 0.35  # Too much selling
        taker_extreme_high = taker_ratio[i] > 0.65  # Too much buying
        
        # Z-score of taker ratio for extreme detection
        taker_zscore = (taker_ratio[i] - taker_ma[i]) / np.where(taker_std[i] == 0, 1e-10, taker_std[i])
        taker_extreme_negative = taker_zscore < -1.2
        taker_extreme_positive = taker_zscore > 1.2
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        if i > 100:
            atr_median = np.nanmedian(atr_14[max(0, i-100):i])
            atr_ratio = atr_14[i] / np.where(atr_median == 0, 1e-10, atr_median)
            vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        else:
            vol_adjustment = 1.0
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: HTF bullish + price at BB lower + RSI oversold OR taker extreme
        if htf_bullish and at_bb_lower:
            # Need either RSI confirmation OR taker volume confirmation
            if rsi_oversold or taker_extreme_low or taker_extreme_negative:
                new_signal = current_size
        
        # SHORT ENTRY: HTF bearish + price at BB upper + RSI overbought OR taker extreme
        elif htf_bearish and at_bb_upper:
            # Need either RSI confirmation OR taker volume confirmation
            if rsi_overbought or taker_extreme_high or taker_extreme_positive:
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 25 bars (~4 days on 4h), allow weaker entry
        if bars_since_last_trade > 25 and new_signal == 0.0 and not in_position:
            if htf_bullish and rsi_14[i] < 35:
                new_signal = current_size * 0.7
            elif htf_bearish and rsi_14[i] > 65:
                new_signal = -current_size * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === MEAN REVERSION EXIT ===
        # Exit when price returns to middle band (mean)
        mean_reversion_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and close[i] >= bb_mid[i]:
                mean_reversion_exit = True
            if position_side < 0 and close[i] <= bb_mid[i]:
                mean_reversion_exit = True
        
        # === HTF TREND REVERSAL EXIT ===
        htf_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and htf_bearish:
                htf_reversal = True
            if position_side < 0 and htf_bullish:
                htf_reversal = True
        
        # === TIME-BASED EXIT ===
        # Exit after 15 bars (~2.5 days) if no profit
        time_exit = False
        if in_position and bars_since_last_trade > 15:
            if position_side > 0 and close[i] < entry_price * 1.01:
                time_exit = True
            if position_side < 0 and close[i] > entry_price * 0.99:
                time_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or mean_reversion_exit or htf_reversal or time_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals