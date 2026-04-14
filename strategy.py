#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Relative Strength Index (RSI) with 1-day Bollinger Band Width Regime Filter
# Uses RSI(14) extremes for mean reversion signals - oversold/overbought conditions
# Bollinger Band Width (BBW) percentile from 1d determines market regime:
# - High BBW (>70th percentile) = trending (use RSI trend following)
# - Low BBW (<30th percentile) = ranging (use RSI mean reversion)
# This adaptive approach works in both bull and bear markets by adjusting to volatility regimes
# Target: 20-35 trades/year (80-140 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop for regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Bollinger Bands (20, 2) for regime detection
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized bandwidth
    
    # Calculate BBW percentile rank (50-period) for regime classification
    bbw_series = pd.Series(bb_width)
    bbw_percentile = bbw_series.rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align 1d BBW percentile to 6h timeframe
    bbw_percentile_aligned = align_htf_to_ltf(prices, df_1d, bbw_percentile)
    
    # Calculate RSI(14) on 6h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for BBW percentile and RSI
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bbw_percentile_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        bbw_pct = bbw_percentile_aligned[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Determine regime: trending (>70) or ranging (<30)
            if bbw_pct > 70:  # Trending regime - trend following
                # Buy on RSI pullback in uptrend (RSI < 40)
                # Sell on RSI bounce in downtrend (RSI > 60)
                if rsi_val < 40:
                    position = 1
                    signals[i] = position_size
                elif rsi_val > 60:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            elif bbw_pct < 30:  # Ranging regime - mean reversion
                # Buy oversold (RSI < 30)
                # Sell overbought (RSI > 70)
                if rsi_val < 30:
                    position = 1
                    signals[i] = position_size
                elif rsi_val > 70:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:  # Neutral regime (30-70) - no clear signal
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI reaches neutral (50) or opposite extreme
            if rsi_val >= 50 or rsi_val > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI reaches neutral (50) or opposite extreme
            if rsi_val <= 50 or rsi_val < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_RSI_BBW_Regime_Adaptive"
timeframe = "6h"
leverage = 1.0