#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter + volume spike (>2x 20-period average) + ATR trailing stop (2.0x)
# Camarilla pivot levels provide high-probability reversal points; breakout of R3/S3 indicates strong momentum
# 1w EMA50 ensures alignment with weekly trend; works in bull/bear markets
# Volume spike >2x average confirms institutional participation; reduces false breakouts
# ATR-based trailing stop manages risk; discrete position sizing (0.25) minimizes fee churn
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe

name = "1d_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
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
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2.0x 20-period average (stricter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(20, 14, 20, 50)  # warmup for volume MA, ATR, Donchian-like calculation, 1w EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_atr = atr[i]
        
        # Calculate Camarilla pivot levels for today (using previous day's OHLC)
        if i >= 1:
            prev_close = close[i-1]
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_range = prev_high - prev_low
            
            # Camarilla levels
            R3 = prev_close + (prev_range * 1.1 / 4)
            S3 = prev_close - (prev_range * 1.1 / 4)
            
            if position == 0:  # Flat - look for new entries
                # Only trade with volume confirmation and trend filter
                if curr_volume_confirm:
                    # Bullish breakout: price above R3 + price above 1w EMA50
                    if curr_close > R3 and curr_close > curr_ema_50_1w:
                        signals[i] = 0.25
                        position = 1
                        entry_price = curr_close
                        highest_since_entry = curr_close
                    # Bearish breakout: price below S3 + price below 1w EMA50
                    elif curr_close < S3 and curr_close < curr_ema_50_1w:
                        signals[i] = -0.25
                        position = -1
                        entry_price = curr_close
                        lowest_since_entry = curr_close
            
            elif position == 1:  # Long position
                # Track highest high since entry for trailing stop
                highest_since_entry = max(highest_since_entry, curr_high)
                # Exit conditions: price breaks below S3 OR ATR trailing stop hit
                if (curr_close < S3 or 
                    curr_close < highest_since_entry - 2.0 * curr_atr):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            
            elif position == -1:  # Short position
                # Track lowest low since entry for trailing stop
                lowest_since_entry = min(lowest_since_entry, curr_low)
                # Exit conditions: price breaks above R3 OR ATR trailing stop hit
                if (curr_close > R3 or 
                    curr_close > lowest_since_entry + 2.0 * curr_atr):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # Not enough data for Camarilla calculation
            signals[i] = 0.0
    
    return signals