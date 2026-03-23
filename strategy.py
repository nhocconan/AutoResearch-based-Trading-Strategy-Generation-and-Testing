#!/usr/bin/env python3
"""
Experiment #667: 1d Primary + 1w HTF — Regime-Adaptive Mean Reversion + Trend

Hypothesis: Daily timeframe with weekly HTF filter provides optimal signal quality
for BTC/ETH which fail simple trend strategies. Key innovation: regime-adaptive
logic that mean-reverts in choppy markets (ADX<20) and trend-follows in trending
markets (ADX>25), with asymmetric sizing based on HTF bias.

Why this should work:
1. 1d TF = ~20-40 trades/year (sweet spot for fee drag vs signal quality)
2. 1w HMA for macro bias — prevents counter-trend trades in strong trends
3. ADX(14) regime detection: <20 = mean revert, >25 = trend follow
4. RSI(14) with LOOSE thresholds (25/75) to ensure trades trigger on all symbols
5. ATR volatility filter — only trade when vol is elevated (avoid dead zones)
6. Donchian(20) breakout confirmation for trend entries
7. Trailing ATR stoploss (2.5x) to protect capital

Key lessons from 441 failures:
- CRSI strategies generate 0 trades (too strict) — use standard RSI instead
- Choppiness Index alone doesn't work — use ADX for regime
- ALL symbols must have positive Sharpe — asymmetric sizing helps
- Need LOOSE thresholds to ensure trade generation on BTC/ETH

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_adaptive_rsi_donchian_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average — smoother than EMA, less lag."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
        rsi_raw = 100 - (100 / (1 + rs))
        rsi[period:] = rsi_raw[period-1:]
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_adx(high, low, close, period=14):
    """Average Directional Index — trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        adx[period*2-1:] = adx_raw[period*2-1:]
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout detection."""
    n = len(high)
    donchian_upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    return donchian_upper, donchian_lower, donchian_mid

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands with width and percentile."""
    n = len(close)
    bb_mid = np.full(n, np.nan)
    bb_upper = np.full(n, np.nan)
    bb_lower = np.full(n, np.nan)
    bb_width = np.full(n, np.nan)
    bb_pct = np.full(n, np.nan)
    
    if n < period:
        return bb_mid, bb_upper, bb_lower, bb_width, bb_pct
    
    bb_mid = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    bb_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    bb_upper = bb_mid + std_mult * bb_std
    bb_lower = bb_mid - std_mult * bb_std
    
    with np.errstate(divide='ignore', invalid='ignore'):
        bb_width = (bb_upper - bb_lower) / bb_mid
        bb_pct = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    
    return bb_mid, bb_upper, bb_lower, bb_width, bb_pct

def calculate_vol_ratio(atr, period_short=7, period_long=30):
    """ATR ratio for volatility spike detection."""
    n = len(atr)
    vol_ratio = np.full(n, np.nan)
    
    if n < period_long:
        return vol_ratio
    
    atr_short = pd.Series(atr).rolling(window=period_short, min_periods=period_short).mean().values
    atr_long = pd.Series(atr).rolling(window=period_long, min_periods=period_long).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_ratio = atr_short / (atr_long + 1e-10)
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    hma_1d = calculate_hma(close, period=21)
    rsi_1d = calculate_rsi(close, period=14)
    adx_1d = calculate_adx(high, low, close, period=14)
    atr_1d = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    bb_mid, bb_upper, bb_lower, bb_width, bb_pct = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    vol_ratio = calculate_vol_ratio(atr_1d, period_short=7, period_long=30)
    
    # Calculate and align HTF indicators (1w)
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    
    # Asymmetric position sizing based on HTF bias
    SIZE_LONG_BASE = 0.30
    SIZE_SHORT_BASE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d[i]) or np.isnan(rsi_1d[i]):
            continue
        if np.isnan(adx_1d[i]) or np.isnan(atr_1d[i]):
            continue
        if np.isnan(hma_1w_aligned[i]) or atr_1d[i] <= 1e-10:
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(bb_pct[i]) or np.isnan(vol_ratio[i]):
            continue
        
        # === REGIME DETECTION ===
        adx_value = adx_1d[i]
        is_trend_regime = adx_value > 25
        is_chop_regime = adx_value < 20
        
        # Volatility filter — only trade when vol is elevated
        vol_elevated = vol_ratio[i] > 1.3
        
        # === HTF TREND BIAS (1w HMA) ===
        htf_1w_bullish = close[i] > hma_1w_aligned[i]
        htf_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === 1d TREND (HMA) ===
        hma_bullish = close[i] > hma_1d[i]
        hma_bearish = close[i] < hma_1d[i]
        
        # === RSI SIGNALS (LOOSE thresholds for trade generation) ===
        rsi_oversold = rsi_1d[i] < 30
        rsi_overbought = rsi_1d[i] > 70
        rsi_extreme_oversold = rsi_1d[i] < 25
        rsi_extreme_overbought = rsi_1d[i] > 75
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        # === BB POSITION ===
        bb_near_lower = bb_pct[i] < 0.15
        bb_near_upper = bb_pct[i] > 0.85
        
        desired_signal = 0.0
        
        # === REGIME 1: TRENDING (ADX > 25) — Trend Follow ===
        if is_trend_regime and vol_elevated:
            # Long: HTF bullish + 1d HMA bullish + RSI not overbought
            if htf_1w_bullish and hma_bullish:
                if donchian_breakout_long:
                    desired_signal = SIZE_LONG_BASE
                elif rsi_oversold and rsi_1d[i] > 20:
                    # Pullback entry in uptrend
                    desired_signal = SIZE_LONG_BASE
            
            # Short: HTF bearish + 1d HMA bearish + RSI not oversold
            elif htf_1w_bearish and hma_bearish:
                if donchian_breakout_short:
                    desired_signal = -SIZE_SHORT_BASE
                elif rsi_overbought and rsi_1d[i] < 80:
                    # Pullback entry in downtrend
                    desired_signal = -SIZE_SHORT_BASE
        
        # === REGIME 2: CHOPPY (ADX < 20) — Mean Reversion ===
        elif is_chop_regime:
            # Long: RSI oversold + BB near lower + HTF not strongly bearish
            if rsi_extreme_oversold and bb_near_lower:
                if not htf_1w_bearish or rsi_1d[i] < 20:
                    desired_signal = SIZE_LONG_BASE
            # Short: RSI overbought + BB near upper + HTF not strongly bullish
            elif rsi_extreme_overbought and bb_near_upper:
                if not htf_1w_bullish or rsi_1d[i] > 80:
                    desired_signal = -SIZE_SHORT_BASE
        
        # === REGIME 3: TRANSITION (20 <= ADX <= 25) — Mixed ===
        else:
            # Use HMA direction with RSI filter — looser conditions
            if hma_bullish and rsi_1d[i] < 65 and vol_elevated:
                desired_signal = SIZE_LONG_BASE
            elif hma_bearish and rsi_1d[i] > 35 and vol_elevated:
                desired_signal = -SIZE_SHORT_BASE
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HMA still bullish AND RSI not extremely overbought
                if hma_bullish and rsi_1d[i] < 80:
                    desired_signal = SIZE_LONG_BASE
            elif position_side < 0:
                # Hold short if HMA still bearish AND RSI not extremely oversold
                if hma_bearish and rsi_1d[i] > 20:
                    desired_signal = -SIZE_SHORT_BASE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG_BASE
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT_BASE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
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