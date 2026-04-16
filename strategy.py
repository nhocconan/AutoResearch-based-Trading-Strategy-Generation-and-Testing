#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w EMA(50) for trend direction, 1d RSI(14) for mean-reversion entries within the trend, and 1d volume spike filter.
# Long when weekly EMA50 is bullish (price > EMA50), RSI < 30 (oversold), and 1d volume > 1.8x 20-period median volume.
# Short when weekly EMA50 is bearish (price < EMA50), RSI > 70 (overbought), and same volume condition.
# Exit via ATR(14) trailing stop: long exits when price < highest high since entry - 2.5*ATR, short exits when price > lowest low since entry + 2.5*ATR.
# Uses discrete position size 0.25. Target: 30-100 total trades over 4 years (7-25/year).
# Weekly EMA50 captures the primary trend, RSI provides mean-reversion entries in the trend direction,
# volume spike confirms institutional interest, ATR stop reduces whipsaws.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data once before loop for EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w Indicators: EMA(50) ===
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d data for RSI, ATR, volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14 = rsi_14.values
    
    # ATR(14)
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr2.iloc[0] = tr1.iloc[0]
    tr3.iloc[0] = tr1.iloc[0]
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    
    # Volume median(20)
    vol_median_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe (1d)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 14, 14, 20)  # EMA(50), RSI(14), ATR(14), volume median(20)
    
    # Track position state for trailing stops
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(vol_median_aligned[i])):
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # Current values (aligned)
        ema50 = ema_50_aligned[i]
        rsi = rsi_aligned[i]
        atr = atr_aligned[i]
        vol_median = vol_median_aligned[i]
        price = close[i]
        
        # Get current 1d volume for volume spike filter
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        current_vol_1d = vol_1d_aligned[i]
        
        # Volume spike filter: current 1d volume > 1.8x median volume
        volume_spike = current_vol_1d > (vol_median * 1.8)
        
        # Trend and mean-reversion filters
        uptrend = price > ema50
        downtrend = price < ema50
        oversold = rsi < 30
        overbought = rsi > 70
        
        # === EXIT LOGIC (trailing stop) ===
        exit_signal = False
        if position == 1:  # long position
            # Update highest high since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Exit when price drops below highest high - 2.5*ATR
            if price < highest_since_entry - 2.5 * atr:
                exit_signal = True
        elif position == -1:  # short position
            # Update lowest low since entry
            if price < lowest_since_entry:
                lowest_since_entry = price
            # Exit when price rises above lowest low + 2.5*ATR
            if price > lowest_since_entry + 2.5 * atr:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Uptrend, RSI oversold, and volume spike
            if uptrend and oversold and volume_spike:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price  # initialize trailing stop
            
            # SHORT CONDITIONS
            # Downtrend, RSI overbought, and volume spike
            elif downtrend and overbought and volume_spike:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price  # initialize trailing stop
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "1d_EMA50_RSI14_VolumeSpike1.8x_ATRTrail2.5_v1"
timeframe = "1d"
leverage = 1.0