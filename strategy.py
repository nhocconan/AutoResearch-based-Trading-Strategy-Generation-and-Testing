#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal Breakout with 1d volume and ATR regime filter
# - Primary: 12h timeframe for lower frequency and reduced fee drag
# - HTF: 1d for volatility (ATR percentile) and volume confirmation
# - Long: Price breaks above recent Williams bearish fractal (high) + 1d ATR > 30th percentile + volume > 1.2x 20-period MA
# - Short: Price breaks below recent Williams bullish fractal (low) + 1d ATR > 30th percentile + volume > 1.2x 20-period MA
# - Exit: Price crosses 12h EMA(21) in opposite direction (trailing stop)
# - Position sizing: 0.25 (discrete level)
# - Target: 80-150 total trades over 4 years (20-38/year) - within 12h sweet spot
# - Works in bull/bear: Fractals capture swing points; volume/ATR filter avoids low-probability breakouts

name = "12h_1d_williams_fractal_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h OHLCV
    open_12h = prices['open'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams Fractals on 1d (5-bar: high/low with 2 lower/higher on each side)
    # Bearish fractal: high[i] is highest of [i-2, i-1, i, i+1, i+2]
    # Bullish fractal: low[i] is lowest of [i-2, i-1, i, i+1, i+2]
    n_1d = len(high_1d)
    bearish_fractal = np.full(n_1d, np.nan)
    bullish_fractal = np.full(n_1d, np.nan)
    
    for i in range(2, n_1d - 2):
        # Bearish fractal: current high is highest in window
        if (high_1d[i] >= high_1d[i-2] and high_1d[i] >= high_1d[i-1] and 
            high_1d[i] >= high_1d[i+1] and high_1d[i] >= high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        # Bullish fractal: current low is lowest in window
        if (low_1d[i] <= low_1d[i-2] and low_1d[i] <= low_1d[i-1] and 
            low_1d[i] <= low_1d[i+1] and low_1d[i] <= low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
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
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h timeframe with proper delay
    # For fractals: need 2 extra bars for confirmation (Williams fractals require 2 bars after)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 12h EMA(21) for exit signal
    close_12h_series = pd.Series(close_12h)
    ema_21_12h = close_12h_series.ewm(span=21, min_periods=21, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(atr_percentile_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i]) or
            np.isnan(ema_21_12h[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 1d volatility regime: ATR > 30th percentile (avoid extremely low vol)
        vol_regime = atr_percentile_aligned[i] > 30
        
        # Volume confirmation: current 1d volume > 1.2x 20-period MA
        volume_spike = volume_1d[i] > 1.2 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above recent bearish fractal + vol regime + volume spike
            if (close_12h[i] > bearish_fractal_aligned[i] and vol_regime and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below recent bullish fractal + vol regime + volume spike
            elif (close_12h[i] < bullish_fractal_aligned[i] and vol_regime and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price crosses 12h EMA(21) in opposite direction
            if position == 1:  # Long position
                exit_condition = close_12h[i] < ema_21_12h[i]  # Cross below EMA
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = close_12h[i] > ema_21_12h[i]  # Cross above EMA
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals