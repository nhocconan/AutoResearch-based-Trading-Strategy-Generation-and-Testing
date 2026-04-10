#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d trend filter
# - Primary: 1h price breaks above/below Camarilla pivot levels (H3/L3) calculated from prior 4h bar
# - HTF: 4h volume > 1.5x 20-period MA for breakout confirmation (avoids low-volume false breakouts)
# - Regime filter: 1d EMA(50) slope > 0 for uptrend bias, < 0 for downtrend bias (trades with higher timeframe trend)
# - Long: Price breaks above H3 + 4h volume confirmation + 1d uptrend (EMA50 rising)
# - Short: Price breaks below L3 + 4h volume confirmation + 1d downtrend (EMA50 falling)
# - Exit: Price returns to pivot point (mean reversion to equilibrium)
# - Session filter: Only trade 08:00-20:00 UTC to avoid low-liquidity Asian session noise
# - Position sizing: 0.20 (discrete level, balances return/drawdown, minimizes fee churn)
# - Works in bull/bear: Camarilla adapts to volatility, volume filters false signals, 1d EMA ensures trend alignment
# - Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe

name = "1h_4h_1d_camarilla_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Precompute session filter (08:00-20:00 UTC) ONCE before loop
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 25 or len(df_1d) < 55:  # Need enough data for calculations
        return np.zeros(n)
    
    # Pre-compute 1h data
    close_1h = prices['close'].values
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    
    # Pre-compute 4h data
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Pre-compute 1d data
    close_1d = df_1d['close'].values
    
    # Calculate 4h Camarilla pivot levels (H3, L3, pivot) from prior 4h bar
    # H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2, pivot = (high+low+close)/3
    camarilla_h3 = np.full(len(close_4h), np.nan)
    camarilla_l3 = np.full(len(close_4h), np.nan)
    camarilla_pivot = np.full(len(close_4h), np.nan)
    
    for i in range(1, len(close_4h)):
        if not (np.isnan(high_4h[i-1]) or np.isnan(low_4h[i-1]) or np.isnan(close_4h[i-1])):
            high_prev = high_4h[i-1]
            low_prev = low_4h[i-1]
            close_prev = close_4h[i-1]
            camarilla_h3[i] = close_prev + 1.1 * (high_prev - low_prev) / 2
            camarilla_l3[i] = close_prev - 1.1 * (high_prev - low_prev) / 2
            camarilla_pivot[i] = (high_prev + low_prev + close_prev) / 3
    
    # Calculate 4h volume moving average (20-period)
    volume_ma_20_4h = np.full(len(volume_4h), np.nan)
    for i in range(19, len(volume_4h)):
        if not np.isnan(volume_4h[i-19:i+1]).any():
            volume_ma_20_4h[i] = np.mean(volume_4h[i-19:i+1])
    
    # Calculate 1d EMA(50) for trend filter
    # Using pandas ewm for efficiency and correctness
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d EMA(50) slope (trend direction: rising/falling)
    ema_slope_1d = np.full(len(ema_50_1d), np.nan)
    for i in range(1, len(ema_50_1d)):
        if not (np.isnan(ema_50_1d[i]) or np.isnan(ema_50_1d[i-1])):
            ema_slope_1d[i] = ema_50_1d[i] - ema_50_1d[i-1]
    
    # Align all HTF/LTF indicators to 1h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot)
    volume_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20_4h)
    ema_slope_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_slope_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(volume_ma_20_4h_aligned[i]) or 
            np.isnan(ema_slope_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period MA
        volume_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
        volume_confirm = volume_4h_aligned[i] > 1.5 * volume_ma_20_4h_aligned[i]
        
        # Trend filter: EMA slope > 0 for uptrend, < 0 for downtrend
        uptrend = ema_slope_1d_aligned[i] > 0
        downtrend = ema_slope_1d_aligned[i] < 0
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above H3 + volume confirmation + uptrend
            if close_1h[i] > camarilla_h3_aligned[i] and volume_confirm and uptrend:
                position = 1
                signals[i] = 0.20
            # Short entry: Price breaks below L3 + volume confirmation + downtrend
            elif close_1h[i] < camarilla_l3_aligned[i] and volume_confirm and downtrend:
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to pivot point (mean reversion to equilibrium)
            if position == 1:  # Long position
                if close_1h[i] <= camarilla_pivot_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # position == -1 (Short position)
                if close_1h[i] >= camarilla_pivot_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
    
    return signals