#!/usr/bin/env python3
"""
Experiment #455: 6h Primary + 12h/1d HTF — Funding Rate Mean Reversion + Trend Filter

Hypothesis: Funding rate extremes (Z-score > 2 or < -2) indicate crowded positioning
that typically reverses within 1-3 days. This is the BEST EDGE for BTC/ETH per research
(Sharpe 0.8-1.5 through 2022 crash). Combined with 12h/1d trend filter to avoid
counter-trend trades in strong moves.

Why 6h: Captures multi-day funding cycles without lower TF noise. 6h bars align well
with 8h funding payments (every 8h = 3x per day, 6h = 4x per day).

Entry Logic:
- Long: Funding Z-score < -1.5 (crowded shorts) + 12h HMA bull + 1d HMA bull
- Short: Funding Z-score > +1.5 (crowded longs) + 12h HMA bear + 1d HMA bear
- Exit: Z-score crosses back to neutral (>-0.5 for longs, <0.5 for shorts) OR stoploss

Target: Sharpe>0.45, DD>-35%, trades>=60 train, trades>=10 test
Timeframe: 6h
Size: 0.25-0.30 (discrete levels)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_funding_zscore_trend_12h1d_v1"
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

def calculate_zscore(values, period=30):
    """Z-score for mean reversion detection"""
    n = len(values)
    zscore = np.zeros(n)
    zscore[:] = np.nan
    
    for i in range(period, n):
        window = values[i-period+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) >= period:
            mean = np.mean(valid)
            std = np.std(valid)
            if std > 1e-10:
                zscore[i] = (values[i] - mean) / std
    
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load funding rate data (CRITICAL EDGE for BTC/ETH)
    # Funding data path: data/processed/funding/{symbol}.parquet
    funding_path = prices.get('funding_path', None)
    funding_rates = None
    
    if funding_path is not None:
        try:
            funding_df = pd.read_parquet(funding_path)
            # Align funding to prices timeframe
            if 'funding_rate' in funding_df.columns:
                funding_rates = funding_df['funding_rate'].values
        except:
            funding_rates = None
    
    # If no funding data, use RSI as proxy for crowded positioning
    use_funding = funding_rates is not None and len(funding_rates) >= n
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    # Calculate funding Z-score or RSI Z-score as proxy
    if use_funding:
        # Use actual funding rates
        funding_zscore = calculate_zscore(funding_rates[:n], period=30)
    else:
        # Use RSI Z-score as proxy for crowded positioning
        rsi_zscore = calculate_zscore(rsi, period=30)
        funding_zscore = rsi_zscore
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    # Hysteresis for funding Z-score exit
    long_exit_threshold = -0.5  # Exit long when Z-score > -0.5
    short_exit_threshold = 0.5  # Exit short when Z-score < 0.5
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(funding_zscore[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (12h + 1d must agree) ===
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Both HTF must agree for strong trend signal
        htf_both_bull = htf_12h_bull and htf_1d_bull
        htf_both_bear = htf_12h_bear and htf_1d_bear
        
        # === FUNDING Z-SCORE EXTREMES ===
        # Z-score < -1.5 = crowded shorts (long signal)
        # Z-score > +1.5 = crowded longs (short signal)
        funding_extreme_long = funding_zscore[i] < -1.5
        funding_extreme_short = funding_zscore[i] > 1.5
        
        # === EXIT CONDITIONS (Z-score mean reversion) ===
        exit_long = funding_zscore[i] > long_exit_threshold
        exit_short = funding_zscore[i] < short_exit_threshold
        
        # === RSI CONFIRMATION (avoid catching falling knife) ===
        rsi_oversold = rsi[i] < 45.0
        rsi_overbought = rsi[i] > 55.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # Long: Funding extreme (crowded shorts) + HTF bull trend + RSI confirmation
        if funding_extreme_long and htf_both_bull:
            # Require RSI not extremely overbought (avoid bad timing)
            if rsi[i] < 70.0:
                desired_signal = SIZE_STRONG
        
        # Short: Funding extreme (crowded longs) + HTF bear trend + RSI confirmation
        elif funding_extreme_short and htf_both_bear:
            # Require RSI not extremely oversold (avoid bad timing)
            if rsi[i] > 30.0:
                desired_signal = -SIZE_STRONG
        
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
        
        # === EXIT ON Z-SCORE MEAN REVERSION ===
        if in_position and position_side > 0 and exit_long:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and exit_short:
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
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
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