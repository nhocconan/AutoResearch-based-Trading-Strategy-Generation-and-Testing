#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 Breakout + 1d EMA34 Trend Filter + Volume Spike + ATR Trailing Stop
# Long when: price breaks above Camarilla R3 (strong resistance) AND price > 1d EMA34 (uptrend) AND volume > 2.0x 20-period avg volume
# Short when: price breaks below Camarilla S3 (strong support) AND price < 1d EMA34 (downtrend) AND volume > 2.0x 20-period avg volume
# Uses ATR trailing stop to limit drawdown in ranging/volatile markets. Discrete sizing (0.25) minimizes fee churn.
# Works in bull/bear via 1d trend filter (avoid counter-trend) + high volume threshold for breakout validity.
# Timeframe: 4h (primary), HTF: 1d for EMA34 trend and Camarilla levels.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d EMA34 trend and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla pivot levels from prior day
    prior_high = pd.Series(df_1d['high'].values).shift(1).values  # shift to use prior day
    prior_low = pd.Series(df_1d['low'].values).shift(1).values
    prior_close = pd.Series(df_1d['close'].values).shift(1).values
    
    # Camarilla levels: R3 = Close + (High - Low) * 1.1/4, S3 = Close - (High - Low) * 1.1/4
    camarilla_r3 = prior_close + (prior_high - prior_low) * 1.1 / 4.0
    camarilla_s3 = prior_close - (prior_high - prior_low) * 1.1 / 4.0
    
    # Align HTF arrays to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume spike: volume > 2.0x 20-period average (stricter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(34, 20, 14)  # warmup for EMA34, volume MA, ATR
    
    for i in range(start_idx, n):
        # Skip if Camarilla data not available (first bar has no prior day)
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_camarilla_r3 = camarilla_r3_aligned[i]
        curr_camarilla_s3 = camarilla_s3_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr[i] if not np.isnan(atr[i]) else 0.0
        
        # Handle position exits and trailing stops
        if position == 1:  # Long position
            highest_since_entry = max(highest_since_entry, curr_close)
            # Exit conditions:
            # 1. ATR trailing stop: price falls below highest - 2.5*ATR
            # 2. Price falls below Camarilla S3 (mean reversion)
            # 3. Price falls below 1d EMA34 (trend change)
            if (curr_close < highest_since_entry - 2.5 * curr_atr or
                curr_close < curr_camarilla_s3 or
                curr_close < curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            lowest_since_entry = min(lowest_since_entry, curr_close)
            # Exit conditions:
            # 1. ATR trailing stop: price rises above lowest + 2.5*ATR
            # 2. Price rises above Camarilla R3 (mean reversion)
            # 3. Price rises above 1d EMA34 (trend change)
            if (curr_close > lowest_since_entry + 2.5 * curr_atr or
                curr_close > curr_camarilla_r3 or
                curr_close > curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 AND above 1d EMA34 AND volume spike
            if (curr_close > curr_camarilla_r3 and
                curr_close > curr_ema_34_1d and
                curr_volume_spike):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            # Short entry: price breaks below Camarilla S3 AND below 1d EMA34 AND volume spike
            elif (curr_close < curr_camarilla_s3 and
                  curr_close < curr_ema_34_1d and
                  curr_volume_spike):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
    
    return signals