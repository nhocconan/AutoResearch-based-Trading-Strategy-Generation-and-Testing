#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above R3 with 1d uptrend (price > EMA34) and volume spike
# Short when price breaks below S3 with 1d downtrend (price < EMA34) and volume spike
# Uses discrete sizing (0.25) to minimize fee churn, targeting 50-150 total trades over 4 years
# Camarilla levels provide institutional support/resistance; volume confirms institutional participation

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Use previous bar's typical price for Camarilla calculation (no look-ahead)
    prev_typical = np.roll(typical_price, 1)
    prev_typical[0] = np.nan  # First bar has no previous
    
    # Camarilla R3/S3 levels
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # Using previous bar's high/low for calculation
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need volume MA(20) and previous bar data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R3, 1d uptrend, volume spike
            if price > camarilla_r3[i] and price > ema_34_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3, 1d downtrend, volume spike
            elif price < camarilla_s3[i] and price < ema_34_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stop or reversal
            # Exit on stoploss (2*ATR) or price breaks below S3 (reversal signal)
            # Simple ATR approximation using recent range
            if i >= 14:
                atr_approx = np.mean(np.maximum(high[i-13:i+1] - low[i-13:i+1], 
                                              np.maximum(np.abs(high[i-13:i+1] - close[i-14:i]), 
                                                       np.abs(low[i-13:i+1] - close[i-14:i]))))
                stop_loss = close[i-1] - 2.0 * atr_approx  # Use previous close for stop calculation
                if price < stop_loss or price < camarilla_s3[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stop or reversal
            # Exit on stoploss (2*ATR) or price breaks above R3 (reversal signal)
            if i >= 14:
                atr_approx = np.mean(np.maximum(high[i-13:i+1] - low[i-13:i+1], 
                                              np.maximum(np.abs(high[i-13:i+1] - close[i-14:i]), 
                                                       np.abs(low[i-13:i+1] - close[i-14:i]))))
                stop_loss = close[i-1] + 2.0 * atr_approx
                if price > stop_loss or price > camarilla_r3[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation
# Williams Alligator consists of three SMAs (Jaw=13, Teeth=8, Lips=5) shifted forward.
# Long when Lips > Teeth > Jaw (bullish alignment) and price above 1d EMA(34), volume > 2.0x 20-bar average.
# Short when Lips < Teeth < Jaw (bearish alignment) and price below 1d EMA(34), volume > 2.0x 20-bar average.
# Uses 6h timeframe targeting 12-37 trades/year (~50-150 total over 4 years) to minimize fee drag.
# Works in bull markets via Alligator bullish alignment and in bear markets via bearish alignment.

name = "6h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator components (SMAs with forward shift)
    # Jaw: 13-period SMA, shifted 8 bars forward
    # Teeth: 8-period SMA, shifted 5 bars forward  
    # Lips: 5-period SMA, shifted 3 bars forward
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # Volume confirmation: >2.0x 20-bar average volume (strict filter to reduce trades)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 13, 8, 5) + 8  # volume MA(20) + max shift (8)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw_vals[i]) or 
            np.isnan(teeth_vals[i]) or np.isnan(lips_vals[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        curr_lips = lips_vals[i]
        curr_teeth = teeth_vals[i]
        curr_jaw = jaw_vals[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Lips > Teeth > Jaw (bullish alignment), price above 1d EMA34, volume spike
            if curr_lips > curr_teeth and curr_teeth > curr_jaw and price > ema_34_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: Lips < Teeth < Jaw (bearish alignment), price below 1d EMA34, volume spike
            elif curr_lips < curr_teeth and curr_teeth < curr_jaw and price < ema_34_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or bearish alignment
            # ATR-based stoploss: 2.0 * ATR below entry (using 6h ATR)
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price - 2.0 * atr_val
            # Exit on stoploss or bearish alignment (Lips < Teeth < Jaw)
            if price < stop_loss or (curr_lips < curr_teeth and curr_teeth < curr_jaw):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stoploss or bullish alignment
            # ATR-based stoploss: 2.0 * ATR above entry
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price + 2.0 * atr_val
            # Exit on stoploss or bullish alignment (Lips > Teeth > Jaw)
            if price > stop_loss or (curr_lips > curr_teeth and curr_teeth > curr_jaw):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals