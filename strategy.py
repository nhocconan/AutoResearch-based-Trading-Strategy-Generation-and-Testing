#!/usr/bin/env python3
"""
Experiment #045: 1h Primary + 4h/1d HTF — Fisher Transform + HMA Trend + Z-Score

Hypothesis: Lower timeframe strategies fail due to excessive trades and fee drag.
This strategy uses HTF trend filter with 1h entry timing to achieve HTF trade
frequency with 1h execution precision.

Key components:
1. 1d HMA(21) for MAJOR trend bias (only trade WITH 1d trend direction)
2. 4h HMA(21) for INTERMEDIATE trend confirmation
3. Ehlers Fisher Transform(9) for reversal entries (proven in bear markets)
4. Z-score(20) for mean reversion confirmation
5. ATR(14) trailing stoploss at 2.5x
6. Discrete position sizing (0.25, 0.30) to minimize fee churn

Why this should work:
- Fisher Transform catches reversals in bear rallies (mentioned in research)
- 1h entries within 4h/1d trend = fewer trades, better timing
- Z-score filter prevents entering at extremes
- Session filter (8-20 UTC) avoids low-liquidity whipsaws
- Target: 30-60 trades/year on 1h timeframe

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h and 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-60/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_zscore_hma_4h1d_v1"
timeframe = "1h"
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
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - catches reversals in bear/range markets.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize to -1 to +1 range
    3. Apply Fisher transform: 0.5 * ln((1+x)/(1-x))
    
    Long when Fisher crosses above -1.5
    Short when Fisher crosses below +1.5
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Typical price
    typical = (high + low) / 2
    typical_s = pd.Series(typical)
    
    # Normalize to -1 to +1 using Donchian-style range
    highest = typical_s.rolling(window=period, min_periods=period).max()
    lowest = typical_s.rolling(window=period, min_periods=period).min()
    
    # Normalize: (price - lowest) / (highest - lowest) * 2 - 1
    range_val = highest - lowest
    normalized = np.where(range_val > 0, (typical - lowest.values) / range_val.values * 2 - 1, 0)
    normalized = np.clip(normalized, -0.99, 0.99)  # Prevent division by zero in log
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    fisher = np.nan_to_num(fisher, nan=0.0)
    
    # Signal line (1-bar lag of fisher)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion detection."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - sma) / std.replace(0, np.nan)
    zscore = zscore.fillna(0).values
    return zscore

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // (1000 * 3600)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    zscore = calculate_zscore(close, 20)
    
    # Volume SMA for filter
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(zscore[i]):
            continue
        
        if np.isnan(volume_sma[i]) or volume_sma[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only - high liquidity) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === 1D TREND BIAS (MAJOR) ===
        # Price above 1d HMA = bullish bias (prefer longs)
        # Price below 1d HMA = bearish bias (prefer shorts)
        trend_1d_bullish = close[i] > hma_1d_21_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H TREND CONFIRMATION (INTERMEDIATE) ===
        trend_4h_bullish = close[i] > hma_4h_21_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.7 * volume_sma[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_long = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_short = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        
        # === Z-SCORE CONFIRMATION ===
        # Long: zscore < -1.0 (price below mean)
        # Short: zscore > 1.0 (price above mean)
        zscore_long = zscore[i] < -0.8
        zscore_short = zscore[i] > 0.8
        
        # === RSI FILTER (avoid extreme counter-trend) ===
        rsi_ok_long = rsi_14[i] < 65  # Not overbought for longs
        rsi_ok_short = rsi_14[i] > 35  # Not oversold for shorts
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Require: 1d bullish OR 4h bullish (trend alignment)
        # Plus: Fisher reversal + Z-score confirmation + volume
        if trend_1d_bullish or trend_4h_bullish:
            if fisher_long and zscore_long and rsi_ok_long and volume_ok:
                new_signal = current_size
        
        # SHORT ENTRIES
        # Require: 1d bearish OR 4h bearish (trend alignment)
        # Plus: Fisher reversal + Z-score confirmation + volume
        if trend_1d_bearish or trend_4h_bearish:
            if fisher_short and zscore_short and rsi_ok_short and volume_ok:
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 150 bars (~6 days on 1h), allow weaker entry
        if bars_since_last_trade > 150 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and trend_4h_bullish and zscore[i] < -1.5:
                new_signal = current_size * 0.7
            elif trend_1d_bearish and trend_4h_bearish and zscore[i] > 1.5:
                new_signal = -current_size * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_1d_bearish and rsi_14[i] > 70:
                trend_reversal = True
            if position_side < 0 and trend_1d_bullish and rsi_14[i] < 30:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === SESSION EXIT (optional - close position if out of session) ===
        # Commented out to allow holding through off-session
        # if in_position and not in_session:
        #     new_signal = 0.0
        
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