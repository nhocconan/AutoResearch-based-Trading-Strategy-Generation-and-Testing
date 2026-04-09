#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w trend filter and ATR volatility filter
# - Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (13-period EMA on 6h)
# - 1w HTF trend: EMA50 > EMA200 = uptrend, < = downtrend
# - ATR filter: ATR(14) < 0.6 * ATR(50) to avoid high volatility whipsaws
# - Long: Bull Power > 0 AND Bear Power < 0 (both bulls and bears agree on strength) in uptrend
# - Short: Bull Power < 0 AND Bear Power > 0 in downtrend
# - Fixed position size 0.25 to control drawdown
# - Target: 12-30 trades/year on 6h timeframe (48-120 total over 4 years)
# - Works in bull/bear via 1w trend filter and Elder Ray momentum confirmation

name = "6h_1w_elder_ray_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMAs for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1w EMAs to 6h timeframe (wait for completed 1w bar)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Pre-compute 6h indicators
    # Elder Ray components
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # High - EMA13
    bear_power = low - ema_13   # Low - EMA13
    
    # ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr_14[i]) or np.isnan(atr_50[i]) or atr_50[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR(14) < 0.6 * ATR(50) to avoid choppy markets
        vol_filter = atr_14[i] < 0.6 * atr_50[i]
        
        # 1w trend filter
        uptrend = ema_50_1w_aligned[i] > ema_200_1w_aligned[i]
        downtrend = ema_50_1w_aligned[i] < ema_200_1w_aligned[i]
        
        # Elder Ray signals
        bull_strong = bull_power[i] > 0      # Bulls in control (high > EMA13)
        bear_weak = bear_power[i] < 0        # Bears weak (low < EMA13)
        bear_strong = bear_power[i] > 0      # Bears in control (low > EMA13)
        bull_weak = bull_power[i] < 0        # Bulls weak (high < EMA13)
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when Elder Ray turns bearish OR trend changes
            if not (bull_strong and bear_weak) or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when Elder Ray turns bullish OR trend changes
            if not (bear_strong and bull_weak) or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Entry conditions with volatility filter
            if vol_filter:
                # Long: Bulls strong AND Bears weak in uptrend
                if uptrend and bull_strong and bear_weak:
                    position = 1
                    signals[i] = position_size
                # Short: Bears strong AND Bulls weak in downtrend
                elif downtrend and bear_strong and bull_weak:
                    position = -1
                    signals[i] = -position_size
    
    return signals