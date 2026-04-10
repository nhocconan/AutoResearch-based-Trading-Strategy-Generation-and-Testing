#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R mean reversion with 4h trend filter and 1d volume confirmation
# - Long when Williams %R(14) < -80 (extreme oversold) + price > 4h EMA(50) (uptrend) + 1d volume > 1.3x 20-period volume SMA
# - Short when Williams %R(14) > -20 (extreme overbought) + price < 4h EMA(50) (downtrend) + 1d volume > 1.3x 20-period volume SMA
# - Exit: Williams %R crosses back above -50 for long exit, below -50 for short exit
# - Position sizing: 0.20 discrete level
# - Works in bull/bear: mean reversion occurs in all regimes, volume filter ensures institutional participation
# - 4h EMA(50) provides trend context to avoid counter-trend trades in strong moves
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise

name = "1h_4h_1d_williamsr_meanrev_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Williams %R(14) on 1h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.fillna(-50).values  # neutral when undefined
    
    # Calculate 4h EMA(50) for trend filter
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d volume SMA(20) for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(williams_r[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for confirmation (aligned to 1h)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)
        vol_confirm = vol_1d_current[i] > 1.3 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: price relative to 4h EMA(50)
        uptrend = close[i] > ema_4h_aligned[i]
        downtrend = close[i] < ema_4h_aligned[i]
        
        # Williams %R extreme levels
        williams_oversold = williams_r[i] < -80
        williams_overbought = williams_r[i] > -20
        
        # Williams %R exit levels (cross -50)
        williams_long_exit = williams_r[i] > -50
        williams_short_exit = williams_r[i] < -50
        
        # Entry conditions
        long_entry = williams_oversold and uptrend and vol_confirm
        short_entry = williams_overbought and downtrend and vol_confirm
        
        # Exit conditions
        exit_long = williams_long_exit
        exit_short = williams_short_exit
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.20
            elif short_entry:
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
    
    return signals