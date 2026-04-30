#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h Supertrend filter and volume confirmation.
# Uses 4h Supertrend(ATR=10, mult=3) for medium-term trend to avoid whipsaws in ranging markets.
# Volume > 2.2x 20-period average confirms momentum (tight threshold to reduce trade frequency).
# ATR-based stoploss (2.0x) limits drawdown. Session filter (08-20 UTC) reduces noise.
# Designed for low trade frequency (~15-25 trades/year) to minimize fee drag on 1h timeframe.
# Works in bull/bear via Supertrend trend filter + volume confirmation + session filter.
# Entry requires 4h Supertrend alignment + volume spike + Camarilla breakout.

name = "1h_Camarilla_R3S3_Breakout_4hSupertrend_VolumeConfirm_ATRStop_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Supertrend trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Supertrend on 4h data
    hl2_4h = (df_4h['high'].values + df_4h['low'].values) / 2
    atr_4h = np.zeros(len(df_4h))
    tr_4h = np.maximum(np.maximum(df_4h['high'].values[1:] - df_4h['low'].values[1:],
                                  np.abs(df_4h['high'].values[1:] - df_4h['close'].values[:-1])),
                         np.abs(df_4h['low'].values[1:] - df_4h['close'].values[:-1]))
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_4h = pd.Series(tr_4h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    upper_4h = hl2_4h + (3.0 * atr_4h)
    lower_4h = hl2_4h - (3.0 * atr_4h)
    
    supertrend_4h = np.zeros(len(df_4h))
    direction_4h = np.ones(len(df_4h))  # 1 for uptrend, -1 for downtrend
    
    supertrend_4h[0] = upper_4h[0]
    direction_4h[0] = 1
    
    for i in range(1, len(df_4h)):
        if close_4h := df_4h['close'].values[i]:
            pass
        if np.isnan(atr_4h[i]) or np.isnan(upper_4h[i]) or np.isnan(lower_4h[i]):
            supertrend_4h[i] = supertrend_4h[i-1]
            direction_4h[i] = direction_4h[i-1]
        else:
            if df_4h['close'].values[i] <= supertrend_4h[i-1]:
                direction_4h[i] = -1
            else:
                direction_4h[i] = 1
            
            if direction_4h[i] == 1 and direction_4h[i-1] == -1:
                supertrend_4h[i] = lower_4h[i]
            elif direction_4h[i] == -1 and direction_4h[i-1] == 1:
                supertrend_4h[i] = upper_4h[i]
            elif direction_4h[i] == 1:
                supertrend_4h[i] = max(supertrend_4h[i-1], lower_4h[i])
            else:
                supertrend_4h[i] = min(supertrend_4h[i-1], upper_4h[i])
    
    # Align Supertrend direction to 1h timeframe
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, direction_4h)
    
    # Calculate ATR(14) for 1h timeframe stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 100  # warmup for Supertrend and ATR
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(supertrend_4h_aligned[i]) or
            np.isnan(atr[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_supertrend_dir = supertrend_4h_aligned[i]
        curr_atr = atr[i]
        
        # Volume confirmation: volume > 2.2x 20-period average (tight threshold to reduce trades)
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (2.2 * vol_ma_20)
        else:
            volume_confirm = False
        
        # Calculate Camarilla levels for current day using previous day's OHLC
        if i >= 24:  # Need at least 24 hours of data for previous day
            # Get timestamp of current bar
            curr_time = prices.iloc[i]["open_time"]
            # Get start of current day (00:00 UTC)
            curr_day_start = curr_time.replace(hour=0, minute=0, second=0, microsecond=0)
            # Get start of previous day
            prev_day_start = curr_day_start - pd.Timedelta(days=1)
            # Get end of previous day (23:59:59.999 UTC)
            prev_day_end = curr_day_start - pd.Timedelta(microseconds=1)
            
            # Filter prices for previous day
            mask = (prices["open_time"] >= prev_day_start) & (prices["open_time"] <= prev_day_end)
            if mask.any():
                prev_day_data = prices.loc[mask]
                if len(prev_day_data) > 0:
                    prev_high = prev_day_data["high"].max()
                    prev_low = prev_day_data["low"].min()
                    prev_close = prev_day_data["close"].iloc[-1]
                    
                    # Calculate Camarilla levels
                    range_val = prev_high - prev_low
                    if range_val > 0:
                        camarilla_r3 = prev_close + (range_val * 1.1 / 4)  # R3 level
                        camarilla_s3 = prev_close - (range_val * 1.1 / 4)  # S3 level
                    else:
                        camarilla_r3 = curr_close
                        camarilla_s3 = curr_close
                else:
                    camarilla_r3 = curr_close
                    camarilla_s3 = curr_close
            else:
                camarilla_r3 = curr_close
                camarilla_s3 = curr_close
        else:
            camarilla_r3 = curr_close
            camarilla_s3 = curr_close
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3, 4h Supertrend uptrend, volume spike
            if (curr_close > camarilla_r3 and 
                curr_supertrend_dir == 1 and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short: price breaks below Camarilla S3, 4h Supertrend downtrend, volume spike
            elif (curr_close < camarilla_s3 and 
                  curr_supertrend_dir == -1 and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: price breaks below Camarilla S3 OR stoploss hit
            if (curr_close < camarilla_s3 or 
                curr_close < entry_price - 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit conditions: price breaks above Camarilla R3 OR stoploss hit
            if (curr_close > camarilla_r3 or 
                curr_close > entry_price + 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals