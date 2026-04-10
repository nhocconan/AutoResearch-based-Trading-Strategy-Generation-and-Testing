#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla Pivot Reversal with 4h trend filter and session timing
# - Primary: 1h timeframe for entry timing precision
# - HTF: 4h for trend direction (EMA21), 1d for volatility regime (ATR percentile)
# - Long: Price breaks above H3 Camarilla pivot (1h) + 4h EMA21 uptrend + 1d ATR > 30th percentile + 08-20 UTC session
# - Short: Price breaks below L3 Camarilla pivot (1h) + 4h EMA21 downtrend + 1d ATR > 30th percentile + 08-20 UTC session
# - Exit: Price reverts to Camarilla Pivot Point (mean reversion) or opposite H3/L3 break
# - Position sizing: 0.20 (discrete level to minimize fee churn)
# - Target: 60-150 total trades over 4 years (15-37/year) - within 1h sweet spot
# - Works in bull/bear: 4h trend filter avoids counter-trend trades in strong moves, Camarilla pivots capture reversals in ranging markets (2025)

name = "1h_4h_1d_camarilla_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1h OHLCV
    open_1h = prices['open'].values
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    volume_1h = prices['volume'].values
    
    # Pre-compute 4h data for EMA trend
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Pre-compute 1d data for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1h Camarilla Pivot Points (based on previous 1h bar)
    # For 1h timeframe, use previous 1h OHLC
    high_1h_prev = np.roll(high_1h, 1)
    low_1h_prev = np.roll(low_1h, 1)
    close_1h_prev = np.roll(close_1h, 1)
    # Set first value to NaN (no previous bar)
    high_1h_prev[0] = np.nan
    low_1h_prev[0] = np.nan
    close_1h_prev[0] = np.nan
    
    rng_1h = high_1h_prev - low_1h_prev
    h3 = close_1h_prev + 1.25 * rng_1h  # Long entry: break above H3
    l3 = close_1h_prev - 1.25 * rng_1h  # Short entry: break below L3
    h3_exit = close_1h_prev + 1.5 * rng_1h  # H4 equivalent for exit
    l3_exit = close_1h_prev - 1.5 * rng_1h  # L4 equivalent for exit
    pivot_1h = (high_1h_prev + low_1h_prev + close_1h_prev) / 3.0  # Mean reversion exit
    
    # Calculate 4h EMA(21) for trend filter
    close_4h_series = pd.Series(close_4h)
    ema_4h = close_4h_series.ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR percentile rank (using 30-day lookback)
    atr_percentile = pd.Series(atr_1d).rolling(window=30, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid or outside session
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(ema_4h_aligned[i]) or 
            np.isnan(atr_percentile_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 4h trend filter: EMA21 slope (using 3-bar change for responsiveness)
        if i >= 33:  # Need enough history for slope calculation
            ema_slope = (ema_4h_aligned[i] - ema_4h_aligned[i-3]) / 3
            uptrend = ema_slope > 0
            downtrend = ema_slope < 0
        else:
            # Fallback to price vs EMA for early bars
            uptrend = close_1h[i] > ema_4h_aligned[i]
            downtrend = close_1h[i] < ema_4h_aligned[i]
        
        # 1d volatility regime: ATR > 30th percentile (avoid extremely low vol)
        vol_regime = atr_percentile_aligned[i] > 30
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above H3 + 4h uptrend + vol regime
            if (close_1h[i] > h3[i] and uptrend and vol_regime):
                position = 1
                signals[i] = 0.20
            # Short entry: Price breaks below L3 + 4h downtrend + vol regime
            elif (close_1h[i] < l3[i] and downtrend and vol_regime):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price reverts to Pivot Point (mean reversion)
            # 2. Price breaks opposite H3/L3 level (contrarian signal)
            
            if position == 1:  # Long position
                exit_condition = (
                    close_1h[i] < pivot_1h[i] or      # Reverted to pivot
                    close_1h[i] < l3[i]               # Broke below L3 (contrarian)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_1h[i] > pivot_1h[i] or      # Reverted to pivot
                    close_1h[i] > h3[i]               # Broke above H3 (contrarian)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
    
    return signals