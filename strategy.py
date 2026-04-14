#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with weekly donchian breakout + volume confirmation + volatility filter
# Targets: 10-30 trades/year by requiring strong weekly breakout conditions
# Logic: Long when price breaks above weekly donchian high (20) with volume spike and low volatility
#        Short when price breaks below weekly donchian low (20) with volume spike and low volatility
#        Uses daily ATR to filter out high volatility periods where breakouts fail
# Position size: 0.25 to manage drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly Donchian channels (20 period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate rolling max/min for donchian channels
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_high_20 = rolling_max(high_1w, 20)
    donchian_low_20 = rolling_min(low_1w, 20)
    
    # Daily ATR (14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(np.concatenate([[high[0]], high[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(np.concatenate([[low[0]], low[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume moving average (20) for volume spike detection
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Get aligned weekly donchian levels
        donchian_high_i = align_htf_to_ltf(prices, df_1w, donchian_high_20)[i]
        donchian_low_i = align_htf_to_ltf(prices, df_1w, donchian_low_20)[i]
        
        if np.isnan(donchian_high_i) or np.isnan(donchian_low_i) or np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i]):
            continue
        
        # Volume spike (1.5x average volume)
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Low volatility filter (ATR below 20-period average)
        atr_ma_20 = pd.Series(atr_14).ewm(span=20, adjust=False, min_periods=20).mean().values
        atr_ma_i = align_htf_to_ltf(prices, pd.DataFrame({'atr': atr_14}), atr_ma_20)['atr'].values[i]
        low_volatility = atr_14[i] < atr_ma_i
        
        # Long: Price breaks above weekly donchian high + volume spike + low volatility
        if position == 0 and close[i] > donchian_high_i and volume_spike and low_volatility:
            position = 1
            signals[i] = position_size
        # Short: Price breaks below weekly donchian low + volume spike + low volatility
        elif position == 0 and close[i] < donchian_low_i and volume_spike and low_volatility:
            position = -1
            signals[i] = -position_size
        # Exit: Price returns to weekly donchian middle or opposite breakout
        elif position != 0:
            donchian_mid = (donchian_high_i + donchian_low_i) / 2
            if position == 1 and close[i] < donchian_mid:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] > donchian_mid:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume_VolatilityFilter"
timeframe = "1d"
leverage = 1.0