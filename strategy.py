#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with 1w volume confirmation and ATR filter
# - Long when price breaks above 20-period Donchian high (1d) AND 1w volume > 1.3x 20-bar avg AND ATR(14) < 0.02 * close (low volatility)
# - Short when price breaks below 20-period Donchian low (1d) AND 1w volume > 1.3x 20-bar avg AND ATR(14) < 0.02 * close
# - Exit when price crosses 10-period EMA (1d) in opposite direction
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Donchian breakouts capture strong momentum moves; volume confirms institutional participation
# - ATR filter ensures trades occur in normal volatility conditions, avoiding chop
# - EMA exit provides timely mean reversion to the trend

name = "1d_donchian_breakout_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d Donchian channels (20-period)
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1d 10-period EMA for exit
    ema_10 = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Pre-compute 1d ATR(14) for volatility filter
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_14[0:13] = np.nan  # First 13 values invalid
    
    # Pre-compute 1w volume confirmation: > 1.3x 20-period average
    volume_1w = df_1w['volume'].values
    volume_20_avg = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_spike_1w = volume_1w > (1.3 * volume_20_avg)
    vol_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_10[i]) or np.isnan(atr_14[i]) or np.isnan(vol_spike_1w_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volatility filter: only trade when ATR < 2% of price (low volatility environment)
        vol_filter = atr_14[i] < (0.02 * close_1d[i])
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND 1w volume spike AND low volatility
            if (prices['high'].iloc[i] > donchian_high[i] and 
                vol_spike_1w_aligned[i] and 
                vol_filter):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND 1w volume spike AND low volatility
            elif (prices['low'].iloc[i] < donchian_low[i] and 
                  vol_spike_1w_aligned[i] and 
                  vol_filter):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when price crosses 10 EMA
            # Exit when price crosses 10 EMA in opposite direction
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] < ema_10[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] > ema_10[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals