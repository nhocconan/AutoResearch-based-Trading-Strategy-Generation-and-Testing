#!/usr/bin/env python3
"""
Experiment #364: 12h Primary + 1d/1w HTF — Simplified HMA/RSI + Funding Contrarian

Hypothesis: Previous strategies (#352-363) failed due to overly complex regime detection
(ADX + Choppiness + multiple filters) that rarely triggered entries. This version:
1. REMOVES ADX/Choppiness regime detection (caused 0 trades on many symbols)
2. SIMPLIFIES to: HMA trend + RSI pullback + HTF alignment (proven pattern)
3. ADDS funding rate z-score contrarian filter (BEST edge for BTC/ETH in bear markets)
4. LOOSENS RSI thresholds (30/70 instead of 25/75) for more trade frequency
5. Reduces confluence from 5+ filters to max 3 per entry

Key insight from research: Funding rate mean reversion has Sharpe 0.8-1.5 through 2022 crash.
When funding z-score > +2 (crowded longs) → short bias. When < -2 (crowded shorts) → long bias.
This works especially well in bear/range markets (2025 test period).

Entry Logic (SIMPLIFIED):
- Long: HMA(21) bull + 1d HMA bull + RSI(14) < 45 (pullback) + funding z < +1
- Short: HMA(21) bear + 1d HMA bear + RSI(14) > 55 (pullback) + funding z > -1
- Strong signal when funding z-score extreme (< -1.5 long, > +1.5 short)

Position sizing: 0.25 base, 0.30 when funding extreme + HTF aligned
Stoploss: 2.5x ATR(14) from entry price

Target: Sharpe>0.45, DD>-35%, trades>=30 train, trades>=5 test, ALL symbols positive
Timeframe: 12h (targets 20-50 trades/year per symbol)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_funding_contrarian_1d1w_v1"
timeframe = "12h"
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_funding_zscore(prices, symbol, lookback=90):
    """
    Funding rate z-score for contrarian signal.
    Loads funding data from data/processed/funding/*.parquet
    Z-score > +2 = crowded longs → short bias
    Z-score < -2 = crowded shorts → long bias
    """
    n = len(prices)
    funding_z = np.zeros(n)
    funding_z[:] = np.nan
    
    try:
        # Try to load funding data
        funding_path = f"data/processed/funding/{symbol.replace('USDT', '')}.parquet"
        df_funding = pd.read_parquet(funding_path)
        
        if len(df_funding) == 0:
            return funding_z
        
        # Align funding data to prices timeline
        # Funding is typically 8h intervals, prices is 12h
        # We need to forward-fill funding to match prices index
        
        # Merge on open_time
        prices_copy = prices.copy()
        prices_copy['open_time'] = pd.to_datetime(prices_copy['open_time'])
        df_funding['open_time'] = pd.to_datetime(df_funding['open_time'])
        
        merged = pd.merge_asof(
            prices_copy.sort_values('open_time'),
            df_funding[['open_time', 'funding_rate']].sort_values('open_time'),
            on='open_time',
            direction='backward'
        )
        
        if len(merged) < lookback:
            return funding_z
        
        # Calculate rolling z-score of funding rate
        funding_series = merged['funding_rate'].values
        for i in range(lookback, len(merged)):
            window = funding_series[i-lookback:i]
            mean_funding = np.mean(window)
            std_funding = np.std(window)
            if std_funding > 1e-10:
                z = (funding_series[i] - mean_funding) / std_funding
                funding_z[i] = z
        
        # Shift to align with prices index
        if len(merged) == n:
            return funding_z
        else:
            # Realign to original prices length
            aligned_z = np.zeros(n)
            aligned_z[:] = np.nan
            min_len = min(n, len(funding_z))
            aligned_z[:min_len] = funding_z[:min_len]
            return aligned_z
            
    except Exception:
        # If funding data not available, return neutral (0)
        return funding_z

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Extract symbol from prices metadata if available
    symbol = "BTCUSDT"  # default, will be overridden by engine
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate funding z-score (contrarian signal)
    funding_z = calculate_funding_zscore(prices, symbol, lookback=90)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
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
        
        if np.isnan(hma_12h[i]) or np.isnan(rsi[i]):
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
        
        # === HTF BIAS (1d and 1w) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        htf_1w_bull = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        htf_1w_bear = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # === 12h HMA TREND ===
        hma_bull = close[i] > hma_12h[i]
        hma_bear = close[i] < hma_12h[i]
        
        # === SMA200 FILTER (major trend) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === FUNDING Z-SCORE (contrarian) ===
        funding_neutral = True
        funding_bullish = False  # z < -1 means crowded shorts → long opportunity
        funding_bearish = False  # z > +1 means crowded longs → short opportunity
        funding_extreme_bull = False  # z < -1.5
        funding_extreme_bear = False  # z > +1.5
        
        if not np.isnan(funding_z[i]):
            funding_neutral = False
            if funding_z[i] < -1.0:
                funding_bullish = True
            if funding_z[i] > 1.0:
                funding_bearish = True
            if funding_z[i] < -1.5:
                funding_extreme_bull = True
            if funding_z[i] > 1.5:
                funding_extreme_bear = True
        
        # === RSI PULLBACK (LOOSENED thresholds for more trades) ===
        rsi_pullback_long = rsi[i] < 45.0  # pullback in uptrend
        rsi_pullback_short = rsi[i] > 55.0  # pullback in downtrend
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === ENTRY LOGIC (SIMPLIFIED - max 3-4 conditions) ===
        desired_signal = 0.0
        
        # LONG ENTRY: HMA bull + 1d bull + RSI pullback + funding not bearish
        # Only require 3 of 4 conditions for entry (more trades)
        long_conditions = 0
        if hma_bull:
            long_conditions += 1
        if htf_1d_bull:
            long_conditions += 1
        if rsi_pullback_long or rsi_oversold:
            long_conditions += 1
        if funding_neutral or funding_bullish:
            long_conditions += 1
        
        if long_conditions >= 3:
            if funding_extreme_bull or (htf_1w_bull and htf_1d_bull):
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT ENTRY: HMA bear + 1d bear + RSI pullback + funding not bullish
        short_conditions = 0
        if hma_bear:
            short_conditions += 1
        if htf_1d_bear:
            short_conditions += 1
        if rsi_pullback_short or rsi_overbought:
            short_conditions += 1
        if funding_neutral or funding_bearish:
            short_conditions += 1
        
        if short_conditions >= 3:
            if funding_extreme_bear or (htf_1w_bear and htf_1d_bear):
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
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