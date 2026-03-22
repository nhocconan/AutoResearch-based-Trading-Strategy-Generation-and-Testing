#!/usr/bin/env python3
"""
Experiment #126: 1d KAMA + Supertrend + 1w HMA Filter + Funding Rate Contrarian + ATR Stop

Hypothesis: Daily timeframe provides cleaner signals with less noise than lower TFs.
Combining multiple proven edges:
- KAMA(21) adapts to volatility (proven in best strategy #118 Sharpe=0.478)
- Supertrend(10,3) for clear trend direction
- 1w HMA(21) for major trend bias (HTF filter)
- Funding rate z-score for contrarian edge (BTC/ETH specific edge from research)
- ATR(14) trailing stop at 2.0*ATR for risk management

Why 1d might work better:
- Fewer false signals in 2022 crash vs lower TFs
- Funding rate mean reversion works best on daily+ timeframes
- Natural fit for swing trading (hold positions days/weeks)
- Less fee drag from fewer trades

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_supertrend_1w_hma_funding_atr_v1"
timeframe = "1d"
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

def calculate_kama(close, period=21, fast=2, slow=30):
    """Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period + slow:
        return kama
    
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    kama[period] = close[period]
    for i in range(period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama

def calculate_hma(close, period=21):
    """Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    n = len(close)
    supertrend = np.zeros(n)
    supertrend[:] = np.nan
    direction = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    atr = calculate_atr(high, low, close, period)
    
    if n < period:
        return supertrend, direction
    
    hl2 = (high + low) / 2
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend[period] = upper_band[period]
    direction[period] = 1
    
    for i in range(period + 1, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
            continue
        
        # Update bands
        if upper_band[i] < supertrend[i-1] or close[i-1] > supertrend[i-1]:
            supertrend[i] = upper_band[i]
        else:
            supertrend[i] = supertrend[i-1]
        
        if lower_band[i] > supertrend[i-1] or close[i-1] < supertrend[i-1]:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = supertrend[i-1]
        
        # Determine direction
        if close[i] > supertrend[i]:
            direction[i] = 1
        else:
            direction[i] = -1
    
    return supertrend, direction

def calculate_funding_zscore(funding_data, lookback=30):
    """Calculate z-score of funding rate for contrarian signal."""
    if funding_data is None or len(funding_data) < lookback:
        return None
    
    funding = funding_data['funding_rate'].values if 'funding_rate' in funding_data.columns else funding_data.values
    n = len(funding)
    zscore = np.zeros(n)
    zscore[:] = np.nan
    
    for i in range(lookback, n):
        window = funding[i-lookback:i]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            mean = np.mean(valid)
            std = np.std(valid)
            if std > 0:
                zscore[i] = (funding[i] - mean) / std
    
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Try to load funding data for contrarian edge
    funding_zscore = None
    try:
        import os
        symbol = prices.get('symbol', 'BTCUSDT')
        if isinstance(symbol, (list, np.ndarray)):
            symbol = symbol[0] if len(symbol) > 0 else 'BTCUSDT'
        funding_path = f"data/processed/funding/{symbol}.parquet"
        if os.path.exists(funding_path):
            funding_df = pd.read_parquet(funding_path)
            if len(funding_df) > 0:
                # Align funding to prices timeframe
                funding_zscore_raw = calculate_funding_zscore(funding_df, 30)
                if funding_zscore_raw is not None:
                    # Simple alignment - take last available funding values
                    min_len = min(n, len(funding_zscore_raw))
                    funding_zscore = np.zeros(n)
                    funding_zscore[:] = np.nan
                    funding_zscore[-min_len:] = funding_zscore_raw[-min_len:]
    except Exception:
        funding_zscore = None
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, 21)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === KAMA ADAPTIVE TREND ===
        bull_kama = close[i] > kama[i]
        bear_kama = close[i] < kama[i]
        
        # === SUPERTREND DIRECTION ===
        st_bull = st_direction[i] == 1
        st_bear = st_direction[i] == -1
        
        # === FUNDING RATE CONTRARIAN ===
        funding_long_signal = False
        funding_short_signal = False
        
        if funding_zscore is not None and not np.isnan(funding_zscore[i]):
            # Extreme negative funding = crowd too short = long opportunity
            if funding_zscore[i] < -1.5:
                funding_long_signal = True
            # Extreme positive funding = crowd too long = short opportunity
            elif funding_zscore[i] > 1.5:
                funding_short_signal = True
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Strong: 1w bullish + KAMA bullish + Supertrend bullish
        if bull_trend_1w and bull_kama and st_bull:
            new_signal = SIZE_STRONG
            # Boost if funding contrarian confirms
            if funding_long_signal:
                new_signal = SIZE_STRONG
        # Moderate: 1w bullish + (KAMA bullish OR Supertrend bullish)
        elif bull_trend_1w and (bull_kama or st_bull):
            new_signal = SIZE_BASE
        # Weak (ensure trades): KAMA bullish + Supertrend bullish
        elif bull_kama and st_bull:
            new_signal = SIZE_BASE
        # Funding contrarian alone (ensure trades in range markets)
        elif funding_long_signal and bull_trend_1w:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Strong: 1w bearish + KAMA bearish + Supertrend bearish
        if bear_trend_1w and bear_kama and st_bear:
            new_signal = -SIZE_STRONG
            if funding_short_signal:
                new_signal = -SIZE_STRONG
        # Moderate: 1w bearish + (KAMA bearish OR Supertrend bearish)
        elif bear_trend_1w and (bear_kama or st_bear):
            new_signal = -SIZE_BASE
        # Weak (ensure trades): KAMA bearish + Supertrend bearish
        elif bear_kama and st_bear:
            new_signal = -SIZE_BASE
        # Funding contrarian alone
        elif funding_short_signal and bear_trend_1w:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            stoploss_price = highest_close - 2.0 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            stoploss_price = lowest_close + 2.0 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0
        
        # Update position tracking
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals