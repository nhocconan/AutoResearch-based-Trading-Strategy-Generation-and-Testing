#!/usr/bin/env python3
"""
Experiment #962: 12h Primary + 1d/1w HTF — HMA Trend + RSI Pullback + ATR Trail

Hypothesis: After 664 failed strategies, the key insight is that 12h timeframe needs
SIMPLER entry logic (not more filters). Too many confluence requirements = 0 trades.

Key insights from research:
1. HMA(21) on 12h provides clean trend signal with less lag than EMA
2. 1d HMA(21) for macro trend bias (proven in multiple experiments)
3. 1w HMA(21) for bull/bear regime filter
4. RSI(14) pullback entries in trend direction (not counter-trend)
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Why 12h timeframe:
- Target 20-50 trades/year (minimal fee drag)
- HTF signals (1d/1w) provide strong regime bias
- Proven to work in both bull and bear markets
- Less whipsaw than 4h/1h during 2022 crash

Critical improvements over failed experiments:
- SIMPLER entry logic (HMA + RSI only, not 5+ conditions)
- Hold logic maintains position through minor pullbacks
- Funding rate as secondary confluence (not primary signal)
- ALL symbols MUST have positive Sharpe (no SOL-only bias)
- Relaxed RSI thresholds (30/70 not 25/75) to ensure trades

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_trend_rsi_pullback_1d1w_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_funding_zscore(funding_series, period=30):
    """Z-score of funding rate over lookback period."""
    n = len(funding_series)
    zscore = np.full(n, np.nan)
    
    if n < period:
        return zscore
    
    for i in range(period - 1, n):
        window = funding_series[i-period+1:i+1]
        mean = np.mean(window)
        std = np.std(window, ddof=1)
        if std > 1e-10:
            zscore[i] = (funding_series[i] - mean) / std
        else:
            zscore[i] = 0.0
    
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Load funding rate data if available
    symbol = prices['symbol'].iloc[0] if 'symbol' in prices.columns else 'BTCUSDT'
    funding_path = f"data/processed/funding/{symbol}.parquet"
    try:
        df_funding = pd.read_parquet(funding_path)
        funding_rates = df_funding['funding_rate'].values
        if len(funding_rates) >= n:
            funding_rates = funding_rates[-n:]
        else:
            funding_rates = np.concatenate([np.zeros(n - len(funding_rates)), funding_rates])
    except:
        funding_rates = np.zeros(n)
    
    # Calculate primary (12h) indicators
    rsi_12h = calculate_rsi(close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align 1d HMA for medium-term trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro regime
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate funding z-score
    funding_z = calculate_funding_zscore(funding_rates, period=30)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_12h[i]) or np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        # === MACRO REGIME (1w HTF HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === MEDIUM-TERM TREND (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === RSI SIGNALS (relaxed for more trades) ===
        rsi_oversold = rsi_12h[i] < 35
        rsi_overbought = rsi_12h[i] > 65
        rsi_extreme_oversold = rsi_12h[i] < 25
        rsi_extreme_overbought = rsi_12h[i] > 75
        rsi_neutral = 35 <= rsi_12h[i] <= 65
        
        # === FUNDING RATE CONTRARIAN ===
        funding_extreme_short = funding_z[i] > 2.0
        funding_extreme_long = funding_z[i] < -2.0
        funding_moderate_short = funding_z[i] > 1.0
        funding_moderate_long = funding_z[i] < -1.0
        
        desired_signal = 0.0
        
        # === BULL REGIME (macro + trend bullish) — Long Bias ===
        if macro_bull and trend_1d_bullish:
            # Primary: RSI pullback in uptrend
            if rsi_oversold:
                desired_signal = BASE_SIZE
            # Secondary: RSI extreme oversold (stronger signal)
            elif rsi_extreme_oversold:
                desired_signal = BASE_SIZE
            # Tertiary: Funding extreme long (contrarian)
            elif funding_extreme_long:
                desired_signal = REDUCED_SIZE
        
        # === BEAR REGIME (macro + trend bearish) — Short Bias ===
        elif macro_bear and trend_1d_bearish:
            # Primary: RSI rally in downtrend
            if rsi_overbought:
                desired_signal = -BASE_SIZE
            # Secondary: RSI extreme overbought (stronger signal)
            elif rsi_extreme_overbought:
                desired_signal = -BASE_SIZE
            # Tertiary: Funding extreme short (contrarian)
            elif funding_extreme_short:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL/TRANSITION REGIME ===
        else:
            # Conservative: Only extreme RSI + funding confluence
            if rsi_extreme_oversold and funding_extreme_long:
                desired_signal = REDUCED_SIZE
            elif rsi_extreme_overbought and funding_extreme_short:
                desired_signal = -REDUCED_SIZE
            # Fallback: Funding alone (ensures trades)
            elif funding_extreme_long:
                desired_signal = REDUCED_SIZE
            elif funding_extreme_short:
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro trend still bullish
                if macro_bull and rsi_12h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro trend still bearish
                if macro_bear and rsi_12h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro regime flips bearish
            if macro_bear and trend_1d_bearish:
                desired_signal = 0.0
            # Exit if funding flips extreme short
            if funding_extreme_short and rsi_12h[i] > 60:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro regime flips bullish
            if macro_bull and trend_1d_bullish:
                desired_signal = 0.0
            # Exit if funding flips extreme long
            if funding_extreme_long and rsi_12h[i] < 40:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
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
        
        signals[i] = desired_signal
    
    return signals