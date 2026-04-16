#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Fractal breakout with 1w volume spike (2.5x median) and 1d ADX regime filter (ADX > 25)
# Long when price breaks above recent bullish fractal high AND 1w volume > 2.5x 20-period median AND 1d ADX > 25
# Short when price breaks below recent bearish fractal low AND 1w volume > 2.5x 20-period median AND 1d ADX > 25
# Exit when price crosses 50-period EMA (mean reversion to equilibrium)
# Uses discrete position size 0.30 to limit fee drag. Target: 30-100 total trades over 4 years.
# Combines fractal structure breakout with volume confirmation and trend regime filter for robustness in bull/bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data once before loop for ADX trend filter and fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: ADX (14-period) for trend regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(np.maximum(tr1, tr2), tr3)
    tr_1d[0] = np.abs(high_1d[0] - low_1d[0])  # first bar
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    up_move = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    down_move = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM
    tr_14 = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_14 = pd.Series(up_move).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_14 = pd.Series(down_move).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (tr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 1d Indicators: Williams Fractals ===
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n] > high[n+1] and high[n] > high[n+2]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n] < low[n+1] and low[n] < low[n+2]
    n_1d = len(high_1d)
    bearish_fractal = np.full(n_1d, np.nan)
    bullish_fractal = np.full(n_1d, np.nan)
    
    for i in range(2, n_1d - 2):
        # Bearish fractal (sell signal)
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] > high_1d[i-1] and 
            high_1d[i] > high_1d[i+1] and 
            high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        
        # Bullish fractal (buy signal)
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] < low_1d[i-1] and 
            low_1d[i] < low_1d[i+1] and 
            low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Align fractals with 2-bar extra delay for confirmation (Williams fractals need 2 bars to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Get 1w data for volume filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1w Indicators: Volume median (20-period) ===
    volume_1w = df_1w['volume'].values
    vol_median_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).median().values
    vol_median_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_median_20_1w)
    
    # Get 1d data for EMA exit
    df_1d_ema = get_htf_data(prices, '1d')
    if len(df_1d_ema) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: EMA (50-period) for exit ===
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d_ema, ema_50)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20, 14)  # 1d EMA, 1w volume, 1d ADX
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0  # force flat outside session
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_median_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 1w volume (aligned)
        vol_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
        if np.isnan(vol_1w_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1w volume > 2.5x 20-period 1w volume median
        vol_threshold = vol_median_20_1w_aligned[i] * 2.5
        vol_confirm = vol_1w_aligned[i] > vol_threshold
        
        # Regime filter: 1d ADX > 25 (trending market)
        regime_filter = adx_1d_aligned[i] > 25
        
        # Price levels
        price = close[i]
        ema_50_val = ema_50_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price crosses below 50-period EMA (mean reversion)
            if price < ema_50_val:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price crosses above 50-period EMA (mean reversion)
            if price > ema_50_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Price breaks above recent bullish fractal high AND volume confirmation AND trend regime
            if not np.isnan(bullish_fractal_val) and price > bullish_fractal_val and vol_confirm and regime_filter:
                signals[i] = 0.30
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below recent bearish fractal low AND volume confirmation AND trend regime
            elif not np.isnan(bearish_fractal_val) and price < bearish_fractal_val and vol_confirm and regime_filter:
                signals[i] = -0.30
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.30  # maintain position
    
    return signals

name = "1d_WilliamsFractal_Breakout_1wVol2.5x_1dADX25_v1"
timeframe = "1d"
leverage = 1.0