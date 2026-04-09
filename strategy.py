#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Camarilla pivot levels with volume confirmation and ATR-based volatility filter
# Camarilla pivots from weekly data provide structured support/resistance levels
# Long when price breaks above H3 with volume confirmation in normal volatility regime
# Short when price breaks below L3 with volume confirmation in normal volatility regime
# In high volatility regime (ATR > 1.5x ATR_mean), reduce position size to 0.15 to avoid whipsaw
# Uses discrete position sizing (0.0, ±0.15, ±0.25) to target ~10-25 trades/year and minimize fee drag
# Works in bull/bear markets: breakout follows trends, mean reversion at pivots during high volatility

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values if 'volume' in df_1w.columns else np.zeros_like(close_1w)
    
    # Calculate 1w Camarilla pivot levels
    range_1w = high_1w - low_1w
    camarilla_h3 = close_1w + 1.1 * range_1w
    camarilla_l3 = close_1w - 1.1 * range_1w
    camarilla_h4 = close_1w + 1.5 * range_1w
    camarilla_l4 = close_1w - 1.5 * range_1w
    
    # Calculate 1w ATR(14) for volatility regime filter
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    atr_mean_1w = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w average volume (20-period)
    vol_s_1w = pd.Series(volume_1w)
    avg_vol_1w = vol_s_1w.rolling(window=20, min_periods=20).mean().values
    
    # Align 1w indicators to 1d timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    avg_vol_1w_aligned = align_htf_to_ltf(prices, df_1w, avg_vol_1w)
    atr_mean_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_mean_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(avg_vol_1w_aligned[i]) or np.isnan(atr_1w_aligned[i]) or np.isnan(atr_mean_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average weekly volume
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # Volatility regime: normal vs high volatility
        high_volatility = atr_1w_aligned[i] > 1.5 * atr_mean_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions
            if high_volatility:
                # In high volatility, exit if price moves back below H3 (mean reversion)
                if close[i] < camarilla_h3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.15  # Reduced size in high vol
            else:
                # In normal volatility, follow trend: exit if price falls below L3
                if close[i] < camarilla_l3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            if high_volatility:
                # In high volatility, exit if price moves back above L3 (mean reversion)
                if close[i] > camarilla_l3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.15  # Reduced size in high vol
            else:
                # In normal volatility, follow trend: exit if price rises above H3
                if close[i] > camarilla_h3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            # Entry conditions
            if volume_confirmed:
                if not high_volatility:
                    # Normal volatility: breakout strategy
                    if close[i] > camarilla_h3_aligned[i]:
                        position = 1
                        signals[i] = 0.25
                    elif close[i] < camarilla_l3_aligned[i]:
                        position = -1
                        signals[i] = -0.25
                else:
                    # High volatility: mean reversion at extremes
                    if close[i] < camarilla_l3_aligned[i]:
                        position = 1
                        signals[i] = 0.15  # Reduced size in high vol
                    elif close[i] > camarilla_h3_aligned[i]:
                        position = -1
                        signals[i] = -0.15  # Reduced size in high vol
    
    return signals