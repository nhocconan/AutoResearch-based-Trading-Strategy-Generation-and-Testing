# 1d_cci_1w_volume_momentum_v1
# Hypothesis: On daily timeframe, use CCI (Commodity Channel Index) from weekly timeframe for trend identification, combined with volume confirmation on daily.
# Enter long when weekly CCI crosses above +100 AND daily volume > 1.5x 20-day average.
# Enter short when weekly CCI crosses below -100 AND daily volume > 1.5x 20-day average.
# Exit when CCI returns to zero line or volume drops below average.
# Weekly timeframe filter reduces noise and captures sustained momentum; volume confirms institutional participation.
# Target: 15-25 trades/year to minimize fee drag while capturing sustained moves in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_cci_1w_volume_momentum_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for CCI calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    
    # Calculate CCI on weekly: (Typical Price - SMA) / (0.015 * Mean Deviation)
    period = 20
    tp = (w_high + w_low + w_close) / 3.0
    tp_series = pd.Series(tp)
    sma_tp = tp_series.rolling(window=period, min_periods=period).mean()
    mean_dev = tp_series.rolling(window=period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci = (tp - sma_tp.values) / (0.015 * mean_dev.values)
    # Handle division by zero or near-zero mean deviation
    cci = np.where(mean_dev.values == 0, 0, cci)
    
    # Align CCI to daily timeframe
    cci_aligned = align_htf_to_ltf(prices, df_1w, cci)
    
    # Volume filter: daily volume > 1.5x 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = vol_series / vol_ma
    vol_ratio = vol_ratio.fillna(0).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start from 1 for previous value comparison
        # Skip if CCI not available
        if np.isnan(cci_aligned[i]):
            signals[i] = 0.0
            continue
        
        # CCI levels
        cci_above_100 = cci_aligned[i] > 100
        cci_below_neg100 = cci_aligned[i] < -100
        cci_near_zero = np.abs(cci_aligned[i]) < 10  # Near zero line
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 1.5
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when CCI drops below +100
            if not cci_above_100:
                exit_long = True
            # Exit when volume drops below average
            elif vol_ratio[i] < 1.0:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when CCI rises above -100
            if not cci_below_neg100:
                exit_short = True
            # Exit when volume drops below average
            elif vol_ratio[i] < 1.0:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: CCI crosses above +100 AND volume confirmed
            cci_prev = cci_aligned[i-1] if i > 0 else -100
            long_entry = (cci_prev <= 100) and cci_above_100 and vol_confirmed
            
            # Short entry: CCI crosses below -100 AND volume confirmed
            short_entry = (cci_prev >= -100) and cci_below_neg100 and vol_confirmed
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals