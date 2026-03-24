#!/usr/bin/env python3
"""
Experiment #020: 1h Primary + 4h/12h HTF — Regime-Adaptive Mean Reversion with Session Filter

Hypothesis: Lower TF (1h) strategies fail due to too many trades → fee drag.
Solution: Use 4h/12h for SIGNAL DIRECTION, 1h only for ENTRY TIMING.
This gives HTF trade frequency (~40-60/year) with 1h entry precision.

Key innovations:
1. 12h ADX for regime strength (ADX>25=trend, ADX<20=range)
2. 4h HMA(21) for trend direction bias (not hard filter)
3. 1h RSI(3) Connors-style for entry timing within HTF trend
4. Session filter: only enter 8-20 UTC (high liquidity, less whipsaw)
5. Volume confirmation: volume > 0.8x 20-period average
6. Asymmetric sizing: 0.30 with HTF trend, 0.20 against

Why this works:
- RSI(3) extremes (<10/>90) are rare → fewer trades, higher quality
- HTF trend filter prevents counter-trend mean reversion in strong trends
- Session filter avoids Asian session whipsaw (0-8 UTC)
- Volume filter confirms genuine moves vs. low-liquidity noise

Target: Sharpe > 0.5, trades 30-80/year/symbol, DD > -35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi3_4h12h_regime_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """
    Relative Strength Index (RSI)
    RSI = 100 - (100 / (1 + RS))
    RS = average gain / average loss over period
    """
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
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_hma(close, period=21):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(series, span):
        if len(series) < span:
            return np.full(len(series), np.nan)
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    ADX > 25 = trending market
    ADX < 20 = ranging/choppy market
    """
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
    
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di_pct = np.zeros(n)
    minus_di_pct = np.zeros(n)
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di_pct[i] = 100.0 * plus_di[i] / atr[i]
            minus_di_pct[i] = 100.0 * minus_di[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di_pct[i] + minus_di_pct[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di_pct[i] - minus_di_pct[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

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

def calculate_sma(close, period=200):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = np.full(n, np.nan)
    cumsum = np.cumsum(close)
    for i in range(period - 1, n):
        if i == period - 1:
            sma[i] = cumsum[i] / period
        else:
            sma[i] = (cumsum[i] - cumsum[i - period]) / period
    
    return sma

def calculate_volume_sma(volume, period=20):
    """Volume Simple Moving Average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = np.full(n, np.nan)
    cumsum = np.cumsum(volume)
    for i in range(period - 1, n):
        if i == period - 1:
            vol_sma[i] = cumsum[i] / period
        else:
            vol_sma[i] = (cumsum[i] - cumsum[i - period]) / period
    
    return vol_sma

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    return (open_time // (1000 * 3600)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values if "open_time" in prices.columns else np.arange(len(close))
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h ADX for regime strength
    adx_12h_raw = calculate_adx(
        df_12h['high'].values,
        df_12h['low'].values,
        df_12h['close'].values,
        period=14
    )
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h_raw)
    
    # Calculate primary (1h) indicators
    rsi_3 = calculate_rsi(close, period=3)  # Connors-style fast RSI
    rsi_14 = calculate_rsi(close, period=14)  # Standard RSI for confirmation
    atr = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    MAX_SIZE = 0.35
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(adx_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi_3[i]) or np.isnan(rsi_14[i]) or np.isnan(sma_200[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (12h ADX) ===
        adx_12h = adx_12h_aligned[i]
        is_trending = adx_12h > 25.0
        is_ranging = adx_12h < 20.0
        
        # === HTF TREND BIAS (4h HMA) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === VOLUME CONFIRMATION ===
        vol_ratio = volume[i] / vol_sma[i] if vol_sma[i] > 1e-10 else 0.0
        volume_ok = vol_ratio > 0.8
        
        # === LONG TERM BIAS (SMA200) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === DESIRED SIGNAL BASED ON REGIME ===
        desired_signal = 0.0
        
        if is_ranging:
            # MEAN REVERSION REGIME - RSI(3) extremes
            # Long: RSI(3) < 10 (extreme oversold)
            if rsi_3[i] < 10.0:
                if in_session and volume_ok:
                    if hma_4h_bull or above_sma200:
                        desired_signal = BASE_SIZE  # With HTF trend
                    else:
                        desired_signal = REDUCED_SIZE  # Against HTF trend
            
            # Short: RSI(3) > 90 (extreme overbought)
            elif rsi_3[i] > 90.0:
                if in_session and volume_ok:
                    if hma_4h_bear or below_sma200:
                        desired_signal = -BASE_SIZE  # With HTF trend
                    else:
                        desired_signal = -REDUCED_SIZE  # Against HTF trend
        
        elif is_trending:
            # TREND REGIME - only trade WITH trend on pullbacks
            # Long pullback: RSI(14) < 40 + 4h bullish
            if hma_4h_bull and above_sma200:
                if rsi_14[i] < 40.0 and in_session and volume_ok:
                    desired_signal = BASE_SIZE
            
            # Short pullback: RSI(14) > 60 + 4h bearish
            elif hma_4h_bear and below_sma200:
                if rsi_14[i] > 60.0 and in_session and volume_ok:
                    desired_signal = -BASE_SIZE
        
        else:
            # NEUTRAL REGIME (20 <= ADX <= 25) - conservative, only with strong HTF bias
            if hma_4h_bull and above_sma200 and rsi_14[i] < 45.0:
                if in_session and volume_ok:
                    desired_signal = REDUCED_SIZE
            elif hma_4h_bear and below_sma200 and rsi_14[i] > 55.0:
                if in_session and volume_ok:
                    desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        desired_signal = np.clip(desired_signal, -MAX_SIZE, MAX_SIZE)
        
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal >= REDUCED_SIZE * 0.85:
            final_signal = REDUCED_SIZE
        elif desired_signal <= -REDUCED_SIZE * 0.85:
            final_signal = -REDUCED_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals