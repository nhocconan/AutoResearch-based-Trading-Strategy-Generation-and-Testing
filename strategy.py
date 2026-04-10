#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla Pivot Breakout with 4h/1d volume and ATR regime filter
# - Primary: 1h timeframe for entry timing precision
# - HTF: 4h for trend direction (EMA21), 1d for volatility (ATR percentile) and volume confirmation
# - Long: Price breaks above H3 Camarilla pivot + 4h EMA21 uptrend + 1d ATR > 40th percentile + volume > 1.3x 20-period MA
# - Short: Price breaks below L3 Camarilla pivot + 4h EMA21 downtrend + 1d ATR > 40th percentile + volume > 1.3x 20-period MA
# - Exit: Price reverts to Camarilla Pivot Point (mean reversion) or breaks H4/L4
# - Session filter: 08-20 UTC to reduce noise trades
# - Position sizing: 0.20 (discrete level)
# - Target: 60-150 total trades over 4 years (15-37/year) - within 1h sweet spot
# - Works in bull/bear: 4h EMA filter avoids counter-trend trades in strong trends, Camarilla pivots capture mean reversion in ranging markets

name = "1h_4h_1d_camarilla_pivot_v1"
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
    
    # Pre-compute 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1h Camarilla Pivot Points (based on previous 4h)
    # Align 4h OHLC to 1h bars (using previous 4h bar's OHLC)
    high_4h_aligned = align_htf_to_ltf(prices, df_4h, high_4h)
    low_4h_aligned = align_htf_to_ltf(prices, df_4h, low_4h)
    close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
    
    # Calculate Camarilla levels for each 1h bar (using previous 4h OHLC)
    rng = high_4h_aligned - low_4h_aligned
    h3 = close_4h_aligned + 1.1 * rng  # Long entry: break above H3 (tighter for 1h)
    l3 = close_4h_aligned - 1.1 * rng  # Short entry: break below L3 (tighter for 1h)
    h4 = close_4h_aligned + 1.3 * rng   # Long exit: break above H4 (take profit)
    l4 = close_4h_aligned - 1.3 * rng   # Short exit: break below L4 (take profit)
    pivot = (high_4h_aligned + low_4h_aligned + close_4h_aligned) / 3.0  # Mean reversion exit
    
    # Calculate 4h EMA21 for trend filter
    close_4h_series = pd.Series(close_4h)
    ema_4h = close_4h_series.ewm(span=21, adjust=False, min_periods=21).mean().values
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
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(ema_4h_aligned[i]) or 
            np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 1d volatility regime: ATR > 40th percentile (avoid low-vol chop)
        vol_regime = atr_percentile_aligned[i] > 40
        
        # Volume confirmation: current 1d volume > 1.3x 20-period MA
        volume_spike = volume_1d[i] > 1.3 * volume_ma_20_1d_aligned[i]
        
        # 4h trend filter: EMA21 direction
        uptrend = close_4h_aligned[i] > ema_4h_aligned[i]
        downtrend = close_4h_aligned[i] < ema_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above H3 resistance + 4h uptrend + vol regime + volume spike
            if (close_1h[i] > h3[i] and uptrend and vol_regime and volume_spike):
                position = 1
                signals[i] = 0.20
            # Short entry: Price breaks below L3 support + 4h downtrend + vol regime + volume spike
            elif (close_1h[i] < l3[i] and downtrend and vol_regime and volume_spike):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price reverts to Pivot Point (mean reversion)
            # 2. Price breaks opposite H4/L4 level (take profit)
            
            if position == 1:  # Long position
                exit_condition = (
                    close_1h[i] < pivot[i] or  # Reverted to pivot
                    close_1h[i] > h4[i]        # Break above H4 (take profit)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_1h[i] > pivot[i] or  # Reverted to pivot
                    close_1h[i] < l4[i]        # Break below L4 (take profit)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
    
    return signals