#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout + 12h ADX trend filter + volume confirmation
# - Primary signal: Bollinger Band width < 20th percentile (squeeze) + breakout above/below bands
# - Trend filter: 12h ADX > 25 ensures breakout occurs in trending environment
# - Volume confirmation: 6h volume > 1.5x 20-period average volume (avoid low-participation false breakouts)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Squeeze breakouts capture volatility expansion; ADX filter ensures
#   trades align with significant trends, reducing whipsaws in ranging markets

name = "6h_12h_bb_squeeze_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h indicators
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h ADX(14) for trend strength
    # +DI, -DI, DX calculation
    up_move = np.diff(high_12h, prepend=high_12h[0])
    down_move = np.diff(low_12h, prepend=low_12h[0]) * -1  # invert to positive
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # True Range
    tr1 = np.abs(np.diff(high_12h, prepend=high_12h[0]))
    tr2 = np.abs(np.diff(low_12h, prepend=low_12h[0]))
    tr3 = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_14 = wilder_smooth(tr, 14)
    plus_di_14 = 100 * wilder_smooth(plus_dm, 14) / tr_14
    minus_di_14 = 100 * wilder_smooth(minus_dm, 14) / tr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = wilder_smooth(dx, 14)
    
    # Align 12h ADX to 6h timeframe (completed 12h bar only)
    adx_14_aligned = align_htf_to_ltf(prices, df_12h, adx_14)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Bollinger Bands(20, 2)
    bb_period = 20
    bb_std = 2.0
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma_20 + (bb_std_dev * bb_std)
    lower_band = sma_20 - (bb_std_dev * bb_std)
    bb_width = (upper_band - lower_band) / sma_20  # normalized width
    
    # 6h BB width percentile (20-period lookback for regime)
    bb_width_percentile = pd.Series(bb_width).rolling(window=40, min_periods=20).rank(pct=True).values
    
    # 6h volume regime: volume > 1.5x 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_regime = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_14_aligned[i]) or
            np.isnan(bb_width_percentile[i]) or
            np.isnan(volume_regime[i]) or
            np.isnan(sma_20[i]) or
            np.isnan(upper_band[i]) or
            np.isnan(lower_band[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below middle band (SMA20) OR ADX drops below 20 (trend weakening)
            if close[i] < sma_20[i] or adx_14_aligned[i] < 20.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above middle band (SMA20) OR ADX drops below 20 (trend weakening)
            if close[i] > sma_20[i] or adx_14_aligned[i] < 20.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Bollinger Band squeeze breakout with volume confirmation and ADX filter
            # Squeeze condition: BB width < 20th percentile (low volatility)
            # Breakout: price closes above upper band (long) or below lower band (short)
            # Trend filter: ADX > 25 (significant trend)
            if (bb_width_percentile[i] < 0.20 and  # squeeze
                volume_regime[i] and                # volume confirmation
                adx_14_aligned[i] > 25.0):          # trend filter
                
                if close[i] > upper_band[i]:  # bullish breakout
                    position = 1
                    signals[i] = 0.25
                elif close[i] < lower_band[i]:  # bearish breakout
                    position = -1
                    signals[i] = -0.25
    
    return signals