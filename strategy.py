#!/usr/bin/env python3
"""
Experiment #1091: 4h Primary + 1d HTF — Dual Regime Choppiness + Mean Reversion/Trend

Hypothesis: After 785+ failed experiments, the key insight is REGIME ADAPTATION.
Market conditions change - what works in trends fails in ranges and vice versa.

Strategy Logic:
1. Choppiness Index (CHOP) determines regime:
   - CHOP > 61.8 = RANGING → Mean reversion at Bollinger Bands
   - CHOP < 38.2 = TRENDING → Donchian breakout following trend
   - Between = NO TRADE (avoid ambiguous regimes)
2. 1d HMA21 for macro bias - only trade in direction of higher TF
3. RSI(14) for entry timing within regime
4. ATR(14) trailing stop 2.5x for risk management
5. ADX(14) > 18 filter for trend confirmation (lower threshold = more trades)

Why this should beat Sharpe=0.612:
- Dual regime adapts to market conditions (proven on ETH Sharpe +0.923)
- Mean reversion works in 2022 crash and 2025 bear market
- Trend following captures 2021 bull and SOL rallies
- 1d HTF filter prevents counter-trend trades
- Looser entry thresholds ensure ≥30 trades/year

Timeframe: 4h (primary)
HTF: 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.25-0.30 discrete levels
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_chop_boll_donchian_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """
    Hull Moving Average — faster and smoother than EMA.
    
    Formula:
    1. WMA(period/2) * 2
    2. WMA(period) * 1
    3. Diff = (1) - (2)
    4. HMA = WMA(sqrt(period)) of Diff
    """
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """
    Relative Strength Index — momentum oscillator.
    """
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index — trend strength indicator.
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = close[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        elif down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    mask = tr_smooth > 1e-10
    plus_di[mask] = 100.0 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100.0 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    di_sum = plus_di + minus_di
    dx = np.zeros(n)
    mask2 = di_sum > 1e-10
    dx[mask2] = 100.0 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market ranging vs trending.
    
    Formula:
    1. ATR(period)
    2. Highest High - Lowest Low over period
    3. CHOP = 100 * log10(sum(ATR) / (HH - LL)) / log10(period)
    
    CHOP > 61.8 = Ranging market
    CHOP < 38.2 = Trending market
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Rolling sum of ATR
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High - Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    range_hl = hh - ll
    
    # CHOP formula
    mask = range_hl > 1e-10
    chop[mask] = 100.0 * np.log10(atr_sum[mask] / range_hl[mask]) / np.log10(period)
    
    return chop

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands for mean reversion entries."""
    n = len(close)
    middle = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n < period:
        return middle, upper, lower
    
    series = pd.Series(close)
    middle = series.rolling(window=period, min_periods=period).mean().values
    std = series.rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    
    return middle, upper, lower

def calculate_donchian(high, low, period=20):
    """Donchian Channel for breakout entries."""
    n = len(close) if 'close' in dir() else len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    return upper, lower, middle

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA21 for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    rsi = calculate_rsi(close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(chop[i]) or np.isnan(bb_mid[i]) or np.isnan(donch_upper[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_ranging = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        # Between 38.2-61.8 = ambiguous, no trade
        
        # === TREND STRENGTH (ADX) ===
        trend_confirmed = adx[i] > 18.0  # Lower threshold for more trades
        
        # === VOLATILITY CHECK ===
        vol_spike = atr[i] > 2.0 * np.nanmedian(atr[max(0, i-100):i]) if i > 100 else False
        current_size = REDUCED_SIZE if vol_spike else BASE_SIZE
        
        desired_signal = 0.0
        
        # === MEAN REVERSION REGIME (Ranging Market) ===
        if is_ranging:
            # Long: Price at lower BB + RSI oversold + macro neutral/bull
            if close[i] <= bb_lower[i] * 1.002 and rsi[i] < 35.0:
                if macro_bull or (not macro_bear):  # Avoid strong bear macro
                    desired_signal = current_size
            
            # Short: Price at upper BB + RSI overbought + macro neutral/bear
            elif close[i] >= bb_upper[i] * 0.998 and rsi[i] > 65.0:
                if macro_bear or (not macro_bull):  # Avoid strong bull macro
                    desired_signal = -current_size
        
        # === TREND FOLLOWING REGIME (Trending Market) ===
        elif is_trending and trend_confirmed:
            # Long: Price breaks Donchian high + RSI rising + macro bull
            if close[i] > donch_upper[i-1] and rsi[i] > 50.0 and rsi[i] < 75.0:
                if macro_bull:
                    desired_signal = current_size
            
            # Short: Price breaks Donchian low + RSI falling + macro bear
            elif close[i] < donch_lower[i-1] and rsi[i] < 50.0 and rsi[i] > 25.0:
                if macro_bear:
                    desired_signal = -current_size
        
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
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if ranging and price above BB mid, or trending and Donchian intact
                if is_ranging and close[i] > bb_mid[i]:
                    desired_signal = current_size
                elif is_trending and close[i] > donch_mid[i] and rsi[i] < 75.0:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if ranging and price below BB mid, or trending and Donchian intact
                if is_ranging and close[i] < bb_mid[i]:
                    desired_signal = -current_size
                elif is_trending and close[i] < donch_mid[i] and rsi[i] > 25.0:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if regime changes or RSI extreme
            if is_trending and rsi[i] > 75.0:
                desired_signal = 0.0
            if is_ranging and close[i] > bb_mid[i] * 1.01 and rsi[i] > 60.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if regime changes or RSI extreme
            if is_trending and rsi[i] < 25.0:
                desired_signal = 0.0
            if is_ranging and close[i] < bb_mid[i] * 0.99 and rsi[i] < 40.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = REDUCED_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = -REDUCED_SIZE * 0.5
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals