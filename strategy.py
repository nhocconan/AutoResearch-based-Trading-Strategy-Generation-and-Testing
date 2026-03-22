#!/usr/bin/env python3
"""
Experiment #349: 15m Multi-Timeframe Mean Reversion with 4h HMA Bias + 1h RSI Filter

Hypothesis: After 297 failed strategies, the pattern is clear:
- Pure trend following fails in bear/range markets (2022 crash, 2025 bear)
- Pure mean reversion fails in strong trends (whipsaw)
- 15m is too noisy alone but works WITH HTF filters

This strategy combines:
1. 4h HMA for trend bias (proven edge in successful strategies)
2. 1h RSI for momentum regime filter (avoid counter-trend in strong trends)
3. 15m Bollinger Bands for mean reversion entries (pullback entries in trend)
4. Volume confirmation (taker_buy_volume ratio) to reduce false signals
5. ATR(14) stoploss at 2.5x to limit drawdown

Why 15m can work:
- Fast enough to catch pullbacks within 4h/1h trends
- With HTF filters, avoids noise and false breakouts
- Mean reversion in trending markets = high win rate entries
- Volume filter reduces churn during low-liquidity periods

Timeframe: 15m (REQUIRED for this experiment)
HTF: 1h and 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels (conservative for 15m noise)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_bb_meanrev_4h_hma_1h_rsi_vol_atr_v1"
timeframe = "15m"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / np.maximum(avg_loss, 1e-10)
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

def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio (0-1, >0.5 = buying pressure)."""
    ratio = taker_buy_volume / np.maximum(volume, 1e-10)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    vol_ratio = calculate_volume_ratio(taker_buy_vol, volume)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25  # Conservative for 15m noise
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === 1h RSI MOMENTUM FILTER ===
        # RSI > 50 = bullish momentum, RSI < 50 = bearish momentum
        rsi_bullish_1h = rsi_1h_aligned[i] > 50
        rsi_bearish_1h = rsi_1h_aligned[i] < 50
        rsi_extreme_bull = rsi_1h_aligned[i] > 70
        rsi_extreme_bear = rsi_1h_aligned[i] < 30
        
        # === VOLUME CONFIRMATION ===
        vol_buying = vol_ratio[i] > 0.55  # Strong buying pressure
        vol_selling = vol_ratio[i] < 0.45  # Strong selling pressure
        
        # === BOLLINGER BAND MEAN REVERSION ===
        # Price near lower band = oversold, near upper = overbought
        bb_width = bb_upper[i] - bb_lower[i]
        bb_position = (close[i] - bb_lower[i]) / np.maximum(bb_width, 1e-10)
        
        # bb_position < 0.1 = near lower band (oversold)
        # bb_position > 0.9 = near upper band (overbought)
        price_near_lower = bb_position < 0.15
        price_near_upper = bb_position > 0.85
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: 4h bullish + 1h RSI not extreme bear + price at BB lower + volume buying
        if bull_trend_4h and rsi_bullish_1h and price_near_lower and vol_buying:
            new_signal = SIZE
        
        # SHORT ENTRY: 4h bearish + 1h RSI not extreme bull + price at BB upper + volume selling
        elif bear_trend_4h and rsi_bearish_1h and price_near_upper and vol_selling:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === 1h RSI EXTREME EXIT ===
        # Exit long if 1h RSI becomes extremely overbought
        if in_position and position_side > 0 and rsi_extreme_bull:
            new_signal = 0.0
        
        # Exit short if 1h RSI becomes extremely oversold
        if in_position and position_side < 0 and rsi_extreme_bear:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals