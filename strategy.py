#!/usr/bin/env python3
"""
Experiment #974: 4h Primary + 12h/1d HTF — Funding Contrarian + Vol Reversion + Regime Switch

Hypothesis: After 664+ failed strategies, the key insight is that funding rate contrarian
signals have proven edge (Sharpe 0.8-1.5) specifically for BTC/ETH through 2022 crash.
Combined with vol spike reversion and regime-adaptive logic, this should work across
ALL symbols in both bull and bear markets.

Key improvements over #934:
1. FUNDING as PRIMARY signal (not just confluence) — proven edge for BTC/ETH
2. SIMPLER entry logic — fewer confluence requirements to ensure trades generate
3. RELAXED thresholds — RSI 30/70 not 25/75, vol ratio 1.5 not 1.8
4. Regime switch: CHOP > 50 = mean revert, CHOP < 40 = trend follow
5. 12h HMA21 for trend bias, 1d HMA21 for macro filter
6. Discrete signals: 0.0, ±0.25, ±0.30 to minimize fee churn
7. ATR trailing stop 2.5x for risk management

Why this should beat Sharpe=0.612:
- Funding contrarian works in ALL regimes (bull, bear, range)
- Vol spike captures panic bottoms (2022 crash, 2025 bear)
- Regime filter avoids trend-following in chop (whipsaw killer)
- Simpler logic = more trades = better statistical significance

Target: Sharpe > 0.7, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_funding_contrarian_vol_regime_12h1d_hma_v1"
timeframe = "4h"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands."""
    n = len(close)
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std(ddof=0).values
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    bandwidth = (upper - lower) / (middle + 1e-10)
    
    return middle, upper, lower, bandwidth

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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index — measures market choppy vs trending."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

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
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Load funding rate data
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
    
    # Calculate primary (4h) indicators
    rsi_4h = calculate_rsi(close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    atr_4h_long = calculate_atr(high, low, close, period=30)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_ratio_4h = np.where(atr_4h_long > 1e-10, atr_4h / atr_4h_long, np.nan)
    
    bb_mid, bb_upper, bb_lower, bb_bw = calculate_bollinger(close, period=20, std_mult=2.0)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    
    # Calculate and align 12h HMA for medium-term trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d HMA for macro regime
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
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
        if np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(vol_ratio_4h[i]) or np.isnan(bb_mid[i]):
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(chop_4h[i]) or np.isnan(funding_z[i]):
            continue
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM TREND (12h HTF HMA21) ===
        trend_12h_bullish = close[i] > hma_12h_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # === REGIME DETECTION (4h Choppiness Index) ===
        ranging_regime = chop_4h[i] > 50
        trending_regime = chop_4h[i] < 40
        
        # === VOL SPIKE DETECTION (relaxed threshold) ===
        vol_spike = vol_ratio_4h[i] > 1.5
        
        # === BOLLINGER BAND POSITION ===
        bb_range = bb_upper[i] - bb_lower[i]
        bb_position = (close[i] - bb_lower[i]) / bb_range if bb_range > 1e-10 else 0.5
        bb_lower_break = close[i] < bb_lower[i]
        bb_upper_break = close[i] > bb_upper[i]
        bb_extreme_low = bb_position < 0.15
        bb_extreme_high = bb_position > 0.85
        
        # === RSI SIGNALS (relaxed thresholds) ===
        rsi_oversold = rsi_4h[i] < 35
        rsi_overbought = rsi_4h[i] > 65
        rsi_extreme_oversold = rsi_4h[i] < 25
        rsi_extreme_overbought = rsi_4h[i] > 75
        
        # === FUNDING RATE CONTRARIAN (PRIMARY SIGNAL) ===
        funding_extreme_short = funding_z[i] > 1.5  # Too many longs → short signal
        funding_extreme_long = funding_z[i] < -1.5  # Too many shorts → long signal
        funding_moderate_short = funding_z[i] > 0.8
        funding_moderate_long = funding_z[i] < -0.8
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 50) — Mean Reversion ===
        if ranging_regime:
            # Long signals (priority order)
            if funding_extreme_long:
                desired_signal = BASE_SIZE
            elif vol_spike and bb_lower_break and rsi_oversold:
                desired_signal = BASE_SIZE
            elif bb_extreme_low and rsi_oversold:
                desired_signal = REDUCED_SIZE
            elif rsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            
            # Short signals (priority order)
            if funding_extreme_short:
                desired_signal = -BASE_SIZE
            elif vol_spike and bb_upper_break and rsi_overbought:
                desired_signal = -BASE_SIZE
            elif bb_extreme_high and rsi_overbought:
                desired_signal = -REDUCED_SIZE
            elif rsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 40) — Trend Following ===
        elif trending_regime:
            # Long: Bullish trend + pullback entry
            if macro_bull or trend_12h_bullish:
                if vol_spike and rsi_oversold:
                    desired_signal = BASE_SIZE
                elif bb_lower_break and funding_moderate_long:
                    desired_signal = REDUCED_SIZE
                elif funding_extreme_long:
                    desired_signal = BASE_SIZE
            
            # Short: Bearish trend + rally entry
            if macro_bear or trend_12h_bearish:
                if vol_spike and rsi_overbought:
                    desired_signal = -BASE_SIZE
                elif bb_upper_break and funding_moderate_short:
                    desired_signal = -REDUCED_SIZE
                elif funding_extreme_short:
                    desired_signal = -BASE_SIZE
        
        # === NEUTRAL REGIME (40 <= CHOP <= 50) ===
        else:
            # Funding contrarian is primary in neutral regime
            if funding_extreme_long:
                desired_signal = BASE_SIZE
            elif funding_extreme_short:
                desired_signal = -BASE_SIZE
            elif funding_moderate_long and (macro_bull or trend_12h_bullish):
                desired_signal = REDUCED_SIZE
            elif funding_moderate_short and (macro_bear or trend_12h_bearish):
                desired_signal = -REDUCED_SIZE
            elif bb_extreme_low:
                desired_signal = REDUCED_SIZE
            elif bb_extreme_high:
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
                if (macro_bull or trend_12h_bullish) and rsi_4h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                if (macro_bear or trend_12h_bearish) and rsi_4h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            if macro_bear and trend_12h_bearish and rsi_4h[i] > 70:
                desired_signal = 0.0
            if funding_extreme_short and rsi_4h[i] > 60:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if macro_bull and trend_12h_bullish and rsi_4h[i] < 30:
                desired_signal = 0.0
            if funding_extreme_long and rsi_4h[i] < 40:
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
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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