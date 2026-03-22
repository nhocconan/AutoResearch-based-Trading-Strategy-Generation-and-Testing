#!/usr/bin/env python3
"""
Experiment #512: 30m Asymmetric Regime-Adaptive with 4h HMA + 1d Volatility Filter

Hypothesis: After 511 failed experiments, the key insight is that 30m timeframe needs:
1. Faster regime detection than daily strategies (30m captures intraday swings)
2. Adaptive RSI thresholds based on rolling percentiles (not fixed 30/70)
3. Bollinger Band position filter for mean-reversion timing
4. Volatility regime from 1d ATR ratio to adjust entry sensitivity
5. Asymmetric logic: bull=buy dips, bear=short rallies (matches BTC/ETH behavior)

Why 30m should work:
- Captures 2-5 day swings (optimal for crypto mean-reversion)
- Less noise than 15m, more signals than 4h
- 4h HMA provides trend bias without daily lag
- 1d ATR ratio detects vol spikes for better entry timing

Key differences from failed strategies:
- RSI percentile ranks (adaptive) vs fixed thresholds
- BB position filter (price within bands) for timing
- Vol regime adjusts sensitivity (looser in high vol)
- Ensures sufficient trades with looser combined filters

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h trend bias + 1d volatility regime via mtf_data helper
Position sizing: 0.25 discrete (conservative for 30m swings)
Stoploss: 2.5 * ATR(14) trailing (tighter than daily)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_asymmetric_4h_hma_1d_vol_bb_rsi_percentile_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

def calculate_rsi_percentile(rsi, lookback=50):
    """Calculate rolling percentile rank of RSI."""
    rsi_s = pd.Series(rsi)
    percentile = rsi_s.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: (x < x.iloc[-1]).sum() / len(x), raw=False
    )
    return percentile.values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if tr_s[i] > 1e-10:
            plus_di = 100 * plus_dm_s[i] / tr_s[i]
            minus_di = 100 * minus_dm_s[i] / tr_s[i]
            di_sum = plus_di + minus_di
            if di_sum > 1e-10:
                dx = 100 * np.abs(plus_di - minus_di) / di_sum
            else:
                dx = 0
        else:
            dx = 0
        
        if i == period:
            adx[i] = dx
        else:
            adx[i] = ((adx[i-1] * (period - 1)) + dx) / period
    
    return adx

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.full(n, np.nan)
    
    close_s = pd.Series(close)
    change = np.abs(close_s.diff(er_period))
    volatility = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    
    er = change / volatility.replace(0, np.inf)
    sc = (er * (2/(fast_period+1) - 2/(slow_period+1)) + 2/(slow_period+1)) ** 2
    
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for trend bias
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d ATR ratio for volatility regime
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_1d_sma = pd.Series(atr_1d).rolling(window=30, min_periods=30).mean().values
    vol_ratio_1d = atr_1d / atr_1d_sma
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    rsi_pct = calculate_rsi_percentile(rsi, 50)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    adx = calculate_adx(high, low, close, 14)
    kama = calculate_kama(close, 10, 2, 30)
    
    # BB position: where price sits within bands (0=lower, 1=upper)
    bb_range = bb_upper - bb_lower
    bb_position = (close - bb_lower) / np.where(bb_range > 1e-10, bb_range, 1)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(rsi_pct[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_position[i]) or np.isnan(vol_ratio_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_regime = close[i] > hma_4h_aligned[i]
        bear_regime = close[i] < hma_4h_aligned[i]
        
        # === 1D VOLATILITY REGIME ===
        high_vol = vol_ratio_aligned[i] > 1.5
        low_vol = vol_ratio_aligned[i] < 0.8
        
        # === ADAPTIVE RSI THRESHOLDS ===
        # Use percentile ranks instead of fixed thresholds
        rsi_oversold = rsi_pct[i] < 0.20  # Bottom 20% of recent RSI
        rsi_overbought = rsi_pct[i] > 0.80  # Top 20% of recent RSI
        
        # === BB POSITION FILTER ===
        bb_low = bb_position[i] < 0.25  # Price in lower quarter of bands
        bb_high = bb_position[i] > 0.75  # Price in upper quarter of bands
        
        # === ASYMMETRIC ENTRY LOGIC ===
        new_signal = 0.0
        
        # BULL REGIME: Favor mean-reversion long (buy dips)
        if bull_regime:
            # Looser thresholds in high vol (more opportunities)
            rsi_threshold = 0.25 if high_vol else 0.20
            bb_threshold = 0.30 if high_vol else 0.25
            
            if rsi_pct[i] < rsi_threshold and bb_position[i] < bb_threshold:
                # Strong mean-reversion signal: oversold RSI + low BB
                new_signal = SIZE
            elif rsi_pct[i] < 0.15 and close[i] > kama[i]:
                # Very oversold + above KAMA (trend intact)
                new_signal = SIZE
            elif bb_position[i] < 0.15 and adx[i] < 25:
                # Extreme BB low + low ADX (ranging, good for mean-rev)
                new_signal = SIZE
        
        # BEAR REGIME: Favor trend-following short (short rallies)
        if bear_regime:
            # Looser thresholds in high vol
            rsi_threshold = 0.75 if high_vol else 0.80
            bb_threshold = 0.70 if high_vol else 0.75
            
            if rsi_pct[i] > rsi_threshold and bb_position[i] > bb_threshold:
                # Strong mean-reversion signal: overbought RSI + high BB
                new_signal = -SIZE
            elif rsi_pct[i] > 0.85 and close[i] < kama[i]:
                # Very overbought + below KAMA (trend down)
                new_signal = -SIZE
            elif bb_position[i] > 0.85 and adx[i] < 25:
                # Extreme BB high + low ADX (ranging, good for mean-rev)
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME REVERSAL EXIT ===
        # Exit if 4h trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_regime:
                new_signal = 0.0
            if position_side < 0 and bull_regime:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals