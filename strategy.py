#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Camarilla pivot levels with volume confirmation and chop regime filter
# In trending regimes (low chop): breakout strategy - long above H3, short below L3 with volume confirmation
# In ranging regimes (high chop): mean reversion - long at L3, short at H3
# Uses discrete position sizing 0.25 to target ~10-30 trades/year on 1d timeframe
# Works in bull/bear markets: breakout captures trends, mean reversion profits from ranging behavior

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
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
    
    # Calculate 1w ATR(14) for stoploss reference
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
    
    # Calculate 1w average volume (20-period)
    vol_s_1w = pd.Series(volume_1w)
    avg_vol_1w = vol_s_1w.rolling(window=20, min_periods=20).mean().values
    
    # Calculate Bollinger Band Width for chop regime filter (using 1w data)
    close_s_1w = pd.Series(close_1w)
    basis_1w = close_s_1w.rolling(window=20, min_periods=20).mean().values
    dev_1w = close_s_1w.rolling(window=20, min_periods=20).std().values
    upper_bb_1w = basis_1w + 2.0 * dev_1w
    lower_bb_1w = basis_1w - 2.0 * dev_1w
    bb_width_1w = (upper_bb_1w - lower_bb_1w) / basis_1w
    bb_width_1w = np.where(basis_1w != 0, bb_width_1w, 0)
    
    # Align 1w indicators to 1d timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    avg_vol_1w_aligned = align_htf_to_ltf(prices, df_1w, avg_vol_1w)
    bb_width_1w_aligned = align_htf_to_ltf(prices, df_1w, bb_width_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(avg_vol_1w_aligned[i]) or np.isnan(bb_width_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # Chop regime: low BB width = trending, high BB width = ranging
        trending_regime = bb_width_1w_aligned[i] < 0.05  # Low volatility = trending
        ranging_regime = bb_width_1w_aligned[i] > 0.10   # High volatility = ranging
        
        if position == 1:  # Long position
            if trending_regime and volume_confirmed:
                # Exit long if price falls below H3
                if close[i] < camarilla_h3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if price moves back above L3 (mean reversion exit)
                if close[i] > camarilla_l3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime and volume_confirmed:
                # Exit short if price rises above L3
                if close[i] > camarilla_l3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if price moves back below H3 (mean reversion exit)
                if close[i] < camarilla_h3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime and volume_confirmed:
                # Breakout strategy in trending market
                if close[i] > camarilla_h3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < camarilla_l3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean reversion at extremes in ranging market
                if close[i] < camarilla_l3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > camarilla_h3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals