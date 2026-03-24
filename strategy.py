#!/usr/bin/env python3
"""
Experiment #323: 6h Primary + 1d/1w HTF — Adaptive Regime + Funding Contrarian + Volume Breakout v1

Hypothesis: 6h timeframe captures multi-day swings without lower-TF noise. Combining:
1. ADAPTIVE REGIME: BB Width percentile + ADX (better than CHOP alone for 6h)
2. FUNDING CONTRARIAN: Z-score < -2 → long, > +2 → short (proven BTC/ETH edge)
3. VOLUME CONFIRMATION: Breakouts require 1.5x avg volume (filters false breakouts)
4. ASYMMETRIC FILTERS: Long only above SMA200, Short only below SMA200
5. HTF ALIGNMENT: 1d HMA for trend, 1w for macro bias

Why 6h: Middle ground between 4h (too noisy) and 12h (too slow). Captures 2-5 day moves.
Target: 30-60 trades/year, Sharpe>0.40, DD>-40%

Key differences from failed 6h experiments:
- NO Fisher Transform (failed in #311, #315, #320)
- NO Weekly Pivot (failed in multiple experiments)
- NO Woodie Pivot (failed in experiments)
- USES volume confirmation + asymmetric SMA200 filter (NEW for 6h)
- STRONGER funding z-score thresholds (-2/+2 not -1.5/+1.5)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_adaptive_regime_funding_vol_breakout_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        if plus_di[i] + minus_di[i] > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma
    
    return upper, lower, bandwidth

def calculate_bb_width_percentile(bandwidth, lookback=100):
    """Percentile rank of BB Width over lookback period"""
    n = len(bandwidth)
    percentile = np.zeros(n)
    percentile[:] = np.nan
    
    for i in range(lookback, n):
        if not np.isnan(bandwidth[i]):
            window = bandwidth[i-lookback:i]
            valid_window = window[~np.isnan(window)]
            if len(valid_window) > 0:
                count_below = np.sum(valid_window < bandwidth[i])
                percentile[i] = 100.0 * count_below / len(valid_window)
    
    return percentile

def calculate_volume_sma(volume, period=20):
    """Volume Simple Moving Average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_funding_zscore(prices, lookback=30):
    """
    Funding Rate Z-Score for contrarian signal
    Load from funding parquet and calculate z-score
    """
    n = len(prices)
    zscore = np.zeros(n)
    zscore[:] = np.nan
    
    try:
        from pathlib import Path
        symbol = "BTCUSDT"
        funding_path = Path("data/processed/funding/BTCUSDT.parquet")
        
        if funding_path.exists():
            funding_df = pd.read_parquet(funding_path)
            if 'funding_rate' in funding_df.columns:
                funding_rates = funding_df['funding_rate'].values
                
                for i in range(lookback, n):
                    if i < len(funding_rates):
                        window = funding_rates[max(0, i-lookback):i]
                        if len(window) >= lookback // 2:
                            mean = np.nanmean(window)
                            std = np.nanstd(window)
                            if std > 1e-10:
                                zscore[i] = (funding_rates[i] - mean) / std
    except Exception:
        pass
    
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    bb_width_pct = calculate_bb_width_percentile(bb_width, lookback=100)
    vol_sma = calculate_volume_sma(volume, period=20)
    sma_200 = calculate_sma(close, 200)
    rsi = calculate_rsi(close, period=14)
    
    # Funding rate z-score (contrarian signal)
    funding_z = calculate_funding_zscore(prices, lookback=30)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Regime memory for hysteresis
    prev_regime = 0  # 0=unknown, 1=trending, 2=range
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_6h[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_width_pct[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === ADAPTIVE REGIME DETECTION ===
        # BB Width percentile < 20 = squeeze (expecting breakout)
        # BB Width percentile > 70 = expansion (range/mean revert)
        # ADX > 25 = trending, ADX < 20 = ranging
        
        bb_squeeze = bb_width_pct[i] < 20.0
        bb_expansion = bb_width_pct[i] > 70.0
        adx_trending = adx[i] > 25.0
        adx_ranging = adx[i] < 20.0
        
        if adx_trending and not bb_expansion:
            current_regime = 1  # trending
        elif adx_ranging or bb_expansion:
            current_regime = 2  # range/mean revert
        elif bb_squeeze:
            current_regime = 3  # squeeze (breakout setup)
        else:
            current_regime = prev_regime
        
        prev_regime = current_regime
        
        # === HTF BIAS ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        htf_1w_bull = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        htf_1w_bear = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === SMA200 ASYMMETRIC FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > 1.5 * vol_sma[i] if not np.isnan(vol_sma[i]) else False
        
        # === RSI EXTREMES ===
        rsi_oversold = not np.isnan(rsi[i]) and rsi[i] < 35.0
        rsi_overbought = not np.isnan(rsi[i]) and rsi[i] > 65.0
        
        # === BOLLINGER BREAKOUT ===
        bb_breakout_long = close[i] > bb_upper[i-1] if not np.isnan(bb_upper[i-1]) else False
        bb_breakout_short = close[i] < bb_lower[i-1] if not np.isnan(bb_lower[i-1]) else False
        
        # === FUNDING RATE CONTRARIAN (STRONG THRESHOLDS) ===
        funding_strong_long = not np.isnan(funding_z[i]) and funding_z[i] < -2.0
        funding_strong_short = not np.isnan(funding_z[i]) and funding_z[i] > 2.0
        funding_moderate_long = not np.isnan(funding_z[i]) and funding_z[i] < -1.0
        funding_moderate_short = not np.isnan(funding_z[i]) and funding_z[i] > 1.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # PRIORITY 1: FUNDING CONTRARIAN (overrides regime)
        if funding_strong_long and above_sma200:
            desired_signal = SIZE_STRONG if htf_1d_bull else SIZE_BASE
        elif funding_strong_short and below_sma200:
            desired_signal = -SIZE_STRONG if htf_1d_bear else -SIZE_BASE
        
        # PRIORITY 2: REGIME 1 - TRENDING (breakout with volume)
        elif current_regime == 1:
            if bb_breakout_long and vol_confirmed and hma_bull and htf_1d_bull and above_sma200:
                desired_signal = SIZE_STRONG if htf_1w_bull else SIZE_BASE
            elif bb_breakout_short and vol_confirmed and hma_bear and htf_1d_bear and below_sma200:
                desired_signal = -SIZE_STRONG if htf_1w_bear else -SIZE_BASE
        
        # PRIORITY 3: REGIME 2 - RANGE (mean reversion with RSI)
        elif current_regime == 2:
            if rsi_oversold and close[i] < bb_lower[i] and above_sma200:
                desired_signal = SIZE_BASE if htf_1d_bull else SIZE_BASE * 0.8
            elif rsi_overbought and close[i] > bb_upper[i] and below_sma200:
                desired_signal = -SIZE_BASE if htf_1d_bear else -SIZE_BASE * 0.8
        
        # PRIORITY 4: REGIME 3 - SQUEEZE BREAKOUT
        elif current_regime == 3:
            if bb_breakout_long and vol_confirmed and htf_1d_bull and above_sma200:
                desired_signal = SIZE_STRONG if htf_1w_bull else SIZE_BASE
            elif bb_breakout_short and vol_confirmed and htf_1d_bear and below_sma200:
                desired_signal = -SIZE_STRONG if htf_1w_bear else -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.8
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals