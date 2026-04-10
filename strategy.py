#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX regime filter + volume confirmation
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# - ADX > 25 indicates trending market (use 1d timeframe for regime)
# - In trending markets (ADX > 25): go long when Bull Power > 0 and rising, short when Bear Power < 0 and falling
# - In ranging markets (ADX <= 25): mean revert at extreme Elder Ray values
# - Volume confirmation: require volume > 1.5x 20-period average
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Elder Ray measures price strength relative to trend (EMA)
# - ADX regime filter ensures we trade with the market structure
# - Works in both bull (trend following) and bear (mean reversion in ranges) markets

name = "6h_1d_elder_ray_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA(13) for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA(13)
    bear_power = low - ema13   # Low - EMA(13)
    
    # Elder Ray momentum (slope)
    bull_power_slope = np.diff(bull_power, prepend=np.nan)
    bear_power_slope = np.diff(bear_power, prepend=np.nan)
    bull_rising = bull_power_slope > 0
    bear_falling = bear_power_slope < 0
    
    # Pre-compute 6h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d ADX for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift(1)).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = -pd.Series(low_1d).diff()
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(arr, period):
        return pd.Series(arr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    period = 14
    atr = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = wilders_smoothing(dx, period)
    
    # ADX > 25 indicates trending market
    adx_trending = adx > 25
    
    # Align HTF indicators to 6h timeframe
    adx_trending_aligned = align_htf_to_ltf(prices, df_1d, adx_trending)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i]) or np.isnan(adx_trending_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            if adx_trending_aligned[i]:  # Trending market
                # Long: Bull Power > 0 and rising
                if bull_power[i] > 0 and bull_rising[i] and volume_spike[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: Bear Power < 0 and falling
                elif bear_power[i] < 0 and bear_falling[i] and volume_spike[i]:
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
            else:  # Ranging market
                # Mean reversion at extreme Elder Ray values
                # Long when Bear Power is extremely negative (oversold)
                # Short when Bull Power is extremely positive (overbought)
                bull_extreme = bull_power[i] > np.percentile(bull_power[max(0, i-100):i+1], 80) if i >= 20 else False
                bear_extreme = bear_power[i] < np.percentile(bear_power[max(0, i-100):i+1], 20) if i >= 20 else False
                
                if bear_extreme and volume_spike[i]:  # Oversold - go long
                    position = 1
                    signals[i] = 0.25
                elif bull_extreme and volume_spike[i]:  # Overbought - go short
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions
            if position == 1:  # Long position
                # Exit long when Bull Power turns negative or loses momentum
                exit_long = bull_power[i] <= 0 or not bull_rising[i]
                # Also exit if we get a short signal in ranging market
                if not adx_trending_aligned[i]:
                    bear_extreme = bear_power[i] < np.percentile(bear_power[max(0, i-100):i+1], 20) if i >= 20 else False
                    exit_long = exit_long or bear_extreme
            else:  # Short position
                # Exit short when Bear Power turns positive or loses momentum
                exit_short = bear_power[i] >= 0 or not bear_falling[i]
                # Also exit if we get a long signal in ranging market
                if not adx_trending_aligned[i]:
                    bull_extreme = bull_power[i] > np.percentile(bull_power[max(0, i-100):i+1], 80) if i >= 20 else False
                    exit_short = exit_short or bull_extreme
            
            if (position == 1 and exit_long) or (position == -1 and exit_short):
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals