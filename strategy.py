# 1d_Camarilla_R3S3_Breakout_1wEMA34_Trend
# Hypothesis: Camarilla R3/S3 breakout on 1d with 1w EMA34 trend filter and volume confirmation.
# Camarilla pivot levels provide strong support/resistance zones; breakouts indicate institutional participation.
# 1w EMA34 filters for long-term trend direction, avoiding counter-trend trades.
# Volume > 2x average confirms breakout validity. Designed for 1d timeframe to minimize trade frequency.
# Works in both bull and bear markets by allowing long/short entries based on trend filter.
# Target: 30-100 total trades over 4 years (7-25/year).
name = "1d_Camarilla_R3S3_Breakout_1wEMA34_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1w close
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels for previous day
    # R3 = H + 1.1*(L - H)/12, S3 = L - 1.1*(H - L)/12
    # Using previous day's high/low to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_r3 = prev_high + 1.1 * (prev_low - prev_high) / 12
    camarilla_s3 = prev_low - 1.1 * (prev_high - prev_low) / 12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1w = ema_34_1w_aligned[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for spike detection
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: Close > R3 AND price > 1w EMA34 (uptrend) AND volume > 2x average
            if close[i] > r3 and close[i] > ema_1w and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Enter short: Close < S3 AND price < 1w EMA34 (downtrend) AND volume > 2x average
            elif close[i] < s3 and close[i] < ema_1w and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close < S3 OR trend reverses (price < 1w EMA34)
            if close[i] < s3 or close[i] < ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close > R3 OR trend reverses (price > 1w EMA34)
            if close[i] > r3 or close[i] > ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals