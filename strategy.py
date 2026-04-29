#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter + volume spike (>2.0x 20-period average) + ATR trailing stop (2.0x)
# Camarilla pivot levels (R3/S3) act as strong intraday support/resistance; breakouts capture momentum with institutional participation
# 1d EMA34 ensures alignment with daily trend; volume spike >2.0x confirms strong participation; ATR-based trailing stop manages risk
# Discrete position sizing (0.25) minimizes fee churn; target 75-200 total trades over 4 years

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR for volatility and stoploss (14-period ATR)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(14, 20, 34)  # warmup for ATR, volume MA, 1d EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(atr[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        
        if i < 2:  # Need at least 2 days for Camarilla calculation
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla levels using previous day's OHLC
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Camarilla pivot levels
        pivot = (prev_high + prev_low + prev_close) / 3
        range_hl = prev_high - prev_low
        
        # R3 and S3 levels (most significant breakout levels)
        r3 = pivot + (range_hl * 1.1 / 2)
        s3 = pivot - (range_hl * 1.1 / 2)
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish breakout: price above R3 + price above 1d EMA34
                if curr_close > r3 and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                # Bearish breakout: price below S3 + price below 1d EMA34
                elif curr_close < s3 and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Track highest high since entry for trailing stop
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit conditions: price breaks below S3 OR ATR trailing stop hit
            if (curr_close < s3 or 
                curr_close < highest_since_entry - 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Track lowest low since entry for trailing stop
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions: price breaks above R3 OR ATR trailing stop hit
            if (curr_close > r3 or 
                curr_close > lowest_since_entry + 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals