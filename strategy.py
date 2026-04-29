#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA50 trend filter and ATR-based volatility regime filter
# Long: Bull Power > 0 AND price > 1d EMA50 AND ATR(14) < ATR(50) (low volatility regime)
# Short: Bear Power < 0 AND price < 1d EMA50 AND ATR(14) < ATR(50) (low volatility regime)
# Exit: Opposite Elder Ray signal OR price crosses 1d EMA50 OR ATR(14) > 1.5 * ATR(50) (high volatility exit)
# Uses 1d HTF for stable trend filter, Elder Ray for momentum, and volatility regime to avoid whipsaws
# Discrete position sizing: 0.25 for long/short, 0.0 for flat to minimize fee churn
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe

name = "6h_ElderRay_1dEMA50_VolRegime_ATRExit_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) and ATR(50) for volatility regime
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Elder Ray Index (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 13, 50)  # warmup for indicators
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_atr_14 = atr_14[i]
        curr_atr_50 = atr_50[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        
        # Volatility regime filter: low volatility (ATR14 < ATR50) for entry
        low_vol_regime = curr_atr_14 < curr_atr_50
        # High volatility exit: ATR14 > 1.5 * ATR50
        high_vol_exit = curr_atr_14 > 1.5 * curr_atr_50
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Exit conditions: Bear Power > 0 (momentum shift) OR price < 1d EMA50 OR high volatility
            if curr_bear_power > 0 or curr_close < curr_ema_1d or high_vol_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: Bull Power < 0 (momentum shift) OR price > 1d EMA50 OR high volatility
            if curr_bull_power < 0 or curr_close > curr_ema_1d or high_vol_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND price > 1d EMA50 AND low volatility regime
            if (curr_bull_power > 0 and 
                curr_close > curr_ema_1d and
                low_vol_regime):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: Bear Power < 0 AND price < 1d EMA50 AND low volatility regime
            elif (curr_bear_power < 0 and 
                  curr_close < curr_ema_1d and
                  low_vol_regime):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals