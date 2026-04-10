#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w HMA trend filter + volume confirmation
# - Primary: 1d timeframe for lower trade frequency (target: 30-100 trades over 4 years)
# - HTF: 1w for HMA(21) trend filter (avoid counter-trend trades)
# - Long: Price breaks above Donchian(20) upper band + HMA(21) 1w > HMA(21) 1w shifted 1 + volume > 1.5x 20-period MA
# - Short: Price breaks below Donchian(20) lower band + HMA(21) 1w < HMA(21) 1w shifted 1 + volume > 1.5x 20-period MA
# - Exit: Price crosses Donchian(20) midpoint (mean reversion) or ATR(14) < 30th percentile (low vol)
# - Position sizing: 0.25 (discrete level)
# - Works in bull/bear: Donchian captures breakouts; HMA filter avoids false signals in wrong trend; volume/vol regime filters reduce whipsaws

name = "1d_1w_donchian_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d OHLCV
    open_1d = prices['open'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    volume_1d = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate 1w HMA(21) for trend filter
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    # Calculate WMA for close_1w
    wma_full = np.full_like(close_1w, np.nan)
    wma_half = np.full_like(close_1w, np.nan)
    
    for i in range(len(close_1w)):
        if i >= 20:  # 21-1 for full window
            wma_full[i] = np.dot(close_1w[i-20:i+1], np.arange(1, 22)) / np.sum(np.arange(1, 22))
        if i >= half_len - 1:
            wma_half[i] = np.dot(close_1w[i-half_len+1:i+1], np.arange(1, half_len+1)) / np.sum(np.arange(1, half_len+1))
    
    # HMA = WMA(2*WMA(half) - WMA(full)), sqrt_len)
    hma_1w = np.full_like(close_1w, np.nan)
    for i in range(len(close_1w)):
        if not np.isnan(wma_full[i]) and not np.isnan(wma_half[i]):
            raw_hma = 2 * wma_half[i] - wma_full[i]
            if i >= sqrt_len - 1:
                hma_1w[i] = np.dot(raw_hma[i-sqrt_len+1:i+1], np.arange(1, sqrt_len+1)) / np.sum(np.arange(1, sqrt_len+1))
    
    # Align 1w HMA to 1d timeframe (wait for completed 1w bar)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1w HMA previous value for trend direction
    hma_1w_prev = np.roll(hma_1w_aligned, 1)
    hma_1w_prev[0] = np.nan
    hma_1w_rising = hma_1w_aligned > hma_1w_prev
    hma_1w_falling = hma_1w_aligned < hma_1w_prev
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR percentile rank (using 30-bar lookback)
    atr_percentile = pd.Series(atr_1d).rolling(window=30, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(hma_1w_aligned[i]) or np.isnan(atr_percentile[i]) or 
            np.isnan(volume_ma_20_1d[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 1d volatility regime: ATR > 30th percentile (avoid extremely low vol)
        vol_regime = atr_percentile[i] > 30
        
        # Volume confirmation: current volume > 1.5x 20-period MA
        volume_spike = volume_1d[i] > 1.5 * volume_ma_20_1d[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above upper Donchian + HMA rising + vol regime + volume spike
            if (close_1d[i] > donchian_upper[i] and hma_1w_rising[i] and vol_regime and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below lower Donchian + HMA falling + vol regime + volume spike
            elif (close_1d[i] < donchian_lower[i] and hma_1w_falling[i] and vol_regime and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price crosses Donchian midpoint (mean reversion)
            # 2. ATR falls below 30th percentile (low volatility regime)
            
            if position == 1:  # Long position
                exit_condition = (
                    close_1d[i] < donchian_mid[i] or  # Price crossed below midpoint
                    atr_percentile[i] < 30  # Low volatility regime
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_1d[i] > donchian_mid[i] or  # Price crossed above midpoint
                    atr_percentile[i] < 30  # Low volatility regime
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals