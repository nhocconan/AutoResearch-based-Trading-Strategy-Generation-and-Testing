#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA(34) trend filter and volume spike confirmation
# Camarilla pivot levels provide strong intraday support/resistance; breakouts beyond R3/S3 indicate momentum
# 1d EMA(34) filters for primary trend alignment (long only above EMA, short only below)
# Volume spike (2.0x 20-period average) confirms institutional participation
# Discrete position sizing 0.25 minimizes fee churn while allowing meaningful exposure
# Targets 20-30 trades/year (80-120 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by requiring volume confirmation and primary trend alignment

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 4h Camarilla levels (R3, S3) from previous day
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # Using previous 1d bar's OHLC for today's levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range / 2
    s3 = prev_close - 1.1 * camarilla_range / 2
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA, Camarilla and volume MA)
    start_idx = 55  # max(34 for EMA, 20 for volume) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3 + price > 1d EMA + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 + price < 1d EMA + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price retreats to Camarilla R4/S4 midpoint (50% retracement)
            # Camarilla: R4 = close + 1.1*(high-low), S4 = close - 1.1*(high-low)
            r4 = prev_close[i] + 1.1 * camarilla_range[i] if not (np.isnan(prev_close[i]) or np.isnan(camarilla_range[i])) else np.nan
            s4 = prev_close[i] - 1.1 * camarilla_range[i] if not (np.isnan(prev_close[i]) or np.isnan(camarilla_range[i])) else np.nan
            midpoint = (r4 + s4) / 2 if not (np.isnan(r4) or np.isnan(s4)) else (r3_aligned[i] + s3_aligned[i]) / 2
            if not np.isnan(midpoint) and close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises to Camarilla R4/S4 midpoint (50% retracement)
            r4 = prev_close[i] + 1.1 * camarilla_range[i] if not (np.isnan(prev_close[i]) or np.isnan(camarilla_range[i])) else np.nan
            s4 = prev_close[i] - 1.1 * camarilla_range[i] if not (np.isnan(prev_close[i]) or np.isnan(camarilla_range[i])) else np.nan
            midpoint = (r4 + s4) / 2 if not (np.isnan(r4) or np.isnan(s4)) else (r3_aligned[i] + s3_aligned[i]) / 2
            if not np.isnan(midpoint) and close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA(34) trend filter and volume spike confirmation
# Camarilla pivot levels provide strong intraday support/resistance; breakouts beyond R3/S3 indicate momentum
# 1d EMA(34) filters for primary trend alignment (long only above EMA, short only below)
# Volume spike (2.0x 20-period average) confirms institutional participation
# Discrete position sizing 0.25 minimizes fee churn while allowing meaningful exposure
# Targets 20-30 trades/year (80-120 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by requiring volume confirmation and primary trend alignment

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 4h Camarilla levels (R3, S3) from previous day
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # Using previous 1d bar's OHLC for today's levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range / 2
    s3 = prev_close - 1.1 * camarilla_range / 2
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA, Camarilla and volume MA)
    start_idx = 55  # max(34 for EMA, 20 for volume) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3 + price > 1d EMA + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 + price < 1d EMA + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price retreats to Camarilla R4/S4 midpoint (50% retracement)
            # Camarilla: R4 = close + 1.1*(high-low), S4 = close - 1.1*(high-low)
            r4 = prev_close[i] + 1.1 * camarilla_range[i] if not (np.isnan(prev_close[i]) or np.isnan(camarilla_range[i])) else np.nan
            s4 = prev_close[i] - 1.1 * camarilla_range[i] if not (np.isnan(prev_close[i]) or np.isnan(camarilla_range[i])) else np.nan
            midpoint = (r4 + s4) / 2 if not (np.isnan(r4) or np.isnan(s4)) else (r3_aligned[i] + s3_aligned[i]) / 2
            if not np.isnan(midpoint) and close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises to Camarilla R4/S4 midpoint (50% retracement)
            r4 = prev_close[i] + 1.1 * camarilla_range[i] if not (np.isnan(prev_close[i]) or np.isnan(camarilla_range[i])) else np.nan
            s4 = prev_close[i] - 1.1 * camarilla_range[i] if not (np.isnan(prev_close[i]) or np.isnan(camarilla_range[i])) else np.nan
            midpoint = (r4 + s4) / 2 if not (np.isnan(r4) or np.isnan(s4)) else (r3_aligned[i] + s3_aligned[i]) / 2
            if not np.isnan(midpoint) and close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals