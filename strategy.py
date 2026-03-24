#!/usr/bin/env python3
"""
Experiment #907: 6h Primary + 1d HTF — Fisher Transform + Funding Z-Score + Regime Adaptive

Hypothesis: 6h timeframe sits between 4h (too noisy) and 12h (too slow). 
Combining Ehlers Fisher Transform (superior reversal detection vs RSI) with 
funding rate z-score (proven BTC/ETH edge) and Choppiness regime filter should 
capture both trend continuations and mean-reversion bounces. 1d HMA provides 
HTF trend bias to avoid counter-trend trades.

Key innovations:
1. Ehlers Fisher Transform(9) - catches reversals 1-2 bars earlier than RSI
2. Funding rate z-score(30) - contrarian edge when funding extreme (>2 or <-2)
3. Choppiness Index(14) regime: >55 = range (mean revert), <45 = trend
4. 1d HMA(21) for HTF bias - only trade with daily trend
5. 6h Donchian(20) breakout confirmation for trend entries
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.25, ±0.30

Entry conditions (LOOSE to ensure ≥30 trades/train, ≥3/test):
- TREND REGIME (CHOP<45): LONG = 1d HMA bull + Fisher>-1.5 + Donchian break up
- TREND REGIME (CHOP<45): SHORT = 1d HMA bear + Fisher<+1.5 + Donchian break down
- RANGE REGIME (CHOP>55): LONG = 1d HMA bull + Fisher<-1.5 OR funding z<-2
- RANGE REGIME (CHOP>55): SHORT = 1d HMA bear + Fisher>+1.5 OR funding z>+2
- FUNDING EXTREME (any regime): z>2.5 → short, z<-2.5 → long (overrides other signals)

Target: Sharpe>0.45, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_funding_chop_regime_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.zeros(n)
    diff[:] = np.nan
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
    return hma

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform
    Converts price to Gaussian distribution for clearer reversal signals
    Fisher = 0.5 * ln((1 + EMA) / (1 - EMA)) + prev_Fisher
    EMA of normalized price (high-low range)
    
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Use high-low midpoint for normalization
    hl_mid = (close + close) / 2  # Simplified: use close for single-series
    
    # Normalize to -1 to +1 range using highest high / lowest low
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(close[i - period + 1:i + 1])
        lowest = np.min(close[i - period + 1:i + 1])
        range_val = highest - lowest
        
        if range_val < 1e-10:
            continue
        
        # Normalize price to -1 to +1
        normalized = 2.0 * (close[i] - lowest) / range_val - 1.0
        
        # Clamp to avoid division issues
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # EMA of normalized value
        if i == period:
            ema_norm = normalized
        else:
            ema_norm = 0.7 * normalized + 0.3 * ema_norm_prev
        
        ema_norm_prev = ema_norm
        
        # Fisher transform
        fisher_val = 0.5 * np.log((1.0 + ema_norm) / (1.0 - ema_norm))
        
        if i > period:
            fisher[i] = 0.7 * fisher_val + 0.3 * fisher[i-1]
            trigger[i] = fisher[i-1]
        else:
            fisher[i] = fisher_val
            trigger[i] = fisher_val
    
    return fisher, trigger

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel
    Upper = highest high over period
    Lower = lowest low over period
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    Using 45/55 thresholds for regime switch
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_zscore(series, period=30):
    """
    Z-Score of series over rolling period
    z = (value - mean) / std
    """
    n = len(series)
    if n < period:
        return np.full(n, np.nan)
    
    zscore = np.zeros(n)
    zscore[:] = np.nan
    
    for i in range(period - 1, n):
        window = series[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        
        if std > 1e-10:
            zscore[i] = (series[i] - mean) / std
        else:
            zscore[i] = 0.0
    
    return zscore

def load_funding_data(symbol):
    """
    Load funding rate data from processed parquet
    Returns array aligned with prices length (filled with NaN if unavailable)
    """
    try:
        # Map symbol to filename
        symbol_map = {
            'BTCUSDT': 'btcusdt',
            'ETHUSDT': 'ethusdt',
            'SOLUSDT': 'solusdt'
        }
        
        base_name = symbol_map.get(symbol, symbol.lower().replace('usdt', ''))
        funding_path = f"data/processed/funding/{base_name}.parquet"
        
        df_funding = pd.read_parquet(funding_path)
        return df_funding['funding_rate'].values
    except:
        return None

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Get symbol from prices metadata if available
    symbol = prices.get('symbol', ['BTCUSDT'])[0] if hasattr(prices, 'get') else 'BTCUSDT'
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Load funding data (optional, for BTC/ETH edge)
    funding_rates = load_funding_data(symbol)
    if funding_rates is not None and len(funding_rates) >= n:
        funding_z = calculate_zscore(funding_rates[:n], period=30)
    else:
        funding_z = np.full(n, np.nan)
    
    # Calculate 6h indicators
    fisher, fisher_trigger = calculate_fisher(close, period=9)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long_cross = False
        fisher_short_cross = False
        
        if i > 0 and not np.isnan(fisher[i-1]) and not np.isnan(fisher_trigger[i-1]):
            # Long: Fisher crosses above -1.5
            fisher_long_cross = (fisher[i-1] <= -1.5) and (fisher[i] > -1.5)
            # Short: Fisher crosses below +1.5
            fisher_short_cross = (fisher[i-1] >= 1.5) and (fisher[i] < 1.5)
        
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # === DONCHIAN BREAKOUT ===
        donchian_break_long = False
        donchian_break_short = False
        
        if not np.isnan(donchian_upper[i]) and not np.isnan(donchian_lower[i]):
            if i > 0:
                donchian_break_long = close[i] > donchian_upper[i-1]
                donchian_break_short = close[i] < donchian_lower[i-1]
        
        # === FUNDING Z-SCORE (contrarian) ===
        funding_extreme_long = False
        funding_extreme_short = False
        
        if not np.isnan(funding_z[i]):
            funding_extreme_long = funding_z[i] < -2.5  # Extremely negative funding → long
            funding_extreme_short = funding_z[i] > 2.5  # Extremely positive funding → short
        
        # === CHOPPINESS REGIME ===
        chop_trending = chop_14[i] < 45.0  # Trend regime
        chop_ranging = chop_14[i] > 55.0  # Range regime
        chop_neutral = not chop_trending and not chop_ranging
        
        # === ENTRY LOGIC (REGIME ADAPTIVE + LOOSE FOR TRADES) ===
        desired_signal = 0.0
        
        # FUNDING EXTREME overrides everything (strong contrarian signal)
        if funding_extreme_long:
            desired_signal = SIZE_STRONG
        elif funding_extreme_short:
            desired_signal = -SIZE_STRONG
        
        elif htf_1d_bull:
            # Bullish HTF bias - prefer longs
            if chop_trending:
                # Trend regime: Fisher cross + Donchian confirmation
                if fisher_long_cross or (fisher_oversold and donchian_break_long):
                    desired_signal = SIZE_STRONG if fisher_long_cross else SIZE_BASE
            elif chop_ranging:
                # Range regime: Fisher mean reversion
                if fisher_oversold or fisher_long_cross:
                    desired_signal = SIZE_STRONG if fisher_long_cross else SIZE_BASE
            else:
                # Neutral regime: loose conditions
                if fisher_oversold or fisher_long_cross or donchian_break_long:
                    desired_signal = SIZE_BASE
        
        elif htf_1d_bear:
            # Bearish HTF bias - prefer shorts
            if chop_trending:
                # Trend regime: Fisher cross + Donchian confirmation
                if fisher_short_cross or (fisher_overbought and donchian_break_short):
                    desired_signal = -SIZE_STRONG if fisher_short_cross else -SIZE_BASE
            elif chop_ranging:
                # Range regime: Fisher mean reversion
                if fisher_overbought or fisher_short_cross:
                    desired_signal = -SIZE_STRONG if fisher_short_cross else -SIZE_BASE
            else:
                # Neutral regime: loose conditions
                if fisher_overbought or fisher_short_cross or donchian_break_short:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
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
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals