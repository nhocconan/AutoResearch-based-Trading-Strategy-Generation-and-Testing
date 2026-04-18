# 6h_Chaikin_Money_Flow_12hTrend_Signal_v1
# Hypothesis: Chaikin Money Flow (CMF) confirms institutional money flow direction.
# Combined with 12h trend filter (EMA34) for higher timeframe bias, this strategy
# captures sustained moves with institutional backing. Works in bull (money inflow
# + uptrend) and bear (money outflow + downtrend). Targets 20-30 trades/year
# to minimize fee drag while capturing high-probability moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Chaikin Money Flow (20-period)
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Avoid division by zero
    hl_range = high - low
    mf_multiplier = np.zeros_like(close)
    mask = hl_range != 0
    mf_multiplier[mask] = ((close[mask] - low[mask]) - (high[mask] - close[mask])) / hl_range[mask]
    
    # Money Flow Volume = MF Multiplier * Volume
    mf_volume = mf_multiplier * volume
    
    # CMF = 20-period sum of MFV / 20-period sum of Volume
    mf_volume_sum = pd.Series(mf_volume).rolling(window=20, min_periods=20).sum().values
    volume_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    cmf = np.zeros_like(close)
    mask_vol = volume_sum != 0
    cmf[mask_vol] = mf_volume_sum[mask_vol] / volume_sum[mask_vol]
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(cmf[i])):
            signals[i] = 0.0
            continue
        
        cmf_val = cmf[i]
        ema_12h = ema_34_12h_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: CMF > 0.05 (buying pressure) + price above 12h EMA34
            if cmf_val > 0.05 and price > ema_12h:
                signals[i] = 0.25
                position = 1
            # Short: CMF < -0.05 (selling pressure) + price below 12h EMA34
            elif cmf_val < -0.05 and price < ema_12h:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: CMF turns negative OR price crosses below EMA
            if cmf_val < 0 or price < ema_12h:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: CMF turns positive OR price crosses above EMA
            if cmf_val > 0 or price > ema_12h:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Chaikin_Money_Flow_12hTrend_Signal_v1"
timeframe = "6h"
leverage = 1.0