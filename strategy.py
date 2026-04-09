#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with 1w trend filter and volume confirmation
# - Primary signal: Daily price breaks above/below 20-day Donchian channel
# - Trend filter: 1-week EMA50 slope (rising/falling) to align with higher timeframe trend
# - Volume confirmation: Daily volume > 20-day median volume (ensures institutional participation)
# - In trending markets (1w EMA50 rising), only take long breakouts; in falling 1w EMA50, only shorts
# - In sideways 1w EMA50 (flat slope), take both directions for mean reversion at extremes
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Stoploss: Exit when price returns to midpoint of Donchian channel
# - Target: 7-25 trades/year (30-100 total over 4 years) per 1d strategy guidelines
# - Works in bull/bear: Donchian captures breakouts, 1w EMA50 filters counter-trend noise

name = "1d_1w_donchian_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d Donchian Channel (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2.0
    
    # 1d volume regime: volume > 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    # Pre-compute 1w indicators
    df_1w_close = df_1w['close'].values
    
    # 1w EMA50 for trend direction
    ema_50_1w = pd.Series(df_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1w EMA50 slope (trend strength): positive = rising, negative = falling, near zero = flat
    # Use 5-period slope to smooth noise
    ema_50_slope = pd.Series(ema_50_1w).diff(5).values / 5.0  # 5-period change per period
    
    # Trend regimes: rising (> 0.001*price), falling (< -0.001*price), flat (otherwise)
    # Scale threshold by price to make it adaptive
    price_level = pd.Series(df_1w_close).rolling(window=50, min_periods=50).mean().values
    threshold = 0.001 * price_level  # 0.1% of price per 5 periods
    w_trend_rising = ema_50_slope > threshold
    w_trend_falling = ema_50_slope < -threshold
    w_trend_flat = np.abs(ema_50_slope) <= threshold
    
    # Align all 1w indicators to 1d timeframe (completed 1w bar only)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    w_trend_rising_aligned = align_htf_to_ltf(prices, df_1w, w_trend_rising)
    w_trend_falling_aligned = align_htf_to_ltf(prices, df_1w, w_trend_falling)
    w_trend_flat_aligned = align_htf_to_ltf(prices, df_1w, w_trend_flat)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(donchian_mid[i]) or
            np.isnan(volume_regime[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(w_trend_rising_aligned[i]) or
            np.isnan(w_trend_falling_aligned[i]) or
            np.isnan(w_trend_flat_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to Donchian midpoint (mean reversion)
            if close[i] <= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to Donchian midpoint (mean reversion)
            if close[i] >= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation
            # Long breakout: price closes above upper Donchian band
            if close[i] > highest_high_20[i] and volume_regime[i]:
                # In rising 1w trend: take long breakouts (trend continuation)
                # In flat 1w trend: take long breakouts (mean reversion setup from extreme)
                # Avoid long in falling 1w trend (counter-trend)
                if w_trend_rising_aligned[i] or w_trend_flat_aligned[i]:
                    position = 1
                    signals[i] = 0.25
            # Short breakout: price closes below lower Donchian band
            elif close[i] < lowest_low_20[i] and volume_regime[i]:
                # In falling 1w trend: take short breakouts (trend continuation)
                # In flat 1w trend: take short breakouts (mean reversion setup from extreme)
                # Avoid short in rising 1w trend (counter-trend)
                if w_trend_falling_aligned[i] or w_trend_flat_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals