#!/usr/bin/env python3
"""
Experiment #184: 12h Primary + 1d/1w HTF — Simplified Trend + Mean Reversion Hybrid

Hypothesis: Previous 12h strategies failed due to TOO MANY filters causing 0 trades.
This version SIMPLIFIES entry logic while keeping proven components:

1. HTF Bias: 1d HMA(50) determines major trend direction
2. Primary Trend: 12h HMA(21) for intermediate trend
3. Entry Trigger: RSI(14) pullback in trend direction (NOT extreme values)
4. Regime Filter: Choppiness Index < 60 to avoid dead chop
5. Volatility Filter: ATR ratio to avoid low-vol traps

Key changes from #172:
- REMOVED Connors RSI (too complex, rare signals)
- REMOVED Donchian breakout (whipsaw in 2022)
- SIMPLIFIED to RSI pullback + HMA trend (proven pattern)
- LOWERED RSI thresholds (30/70 instead of 15/85) for MORE trades
- ADDED funding rate contrarian signal for BTC/ETH edge

Position sizing: 0.25 base, 0.30 strong
Stoploss: 2.5x ATR trailing
Target: Sharpe>0.40 (beat #183), trades>=30 train, trades>=5 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_chop_funding_1d1w_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = choppy/range bound
    CHOP < 38.2 = trending
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
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def load_funding_data(symbol):
    """
    Load funding rate data for contrarian signal.
    Returns z-score of 30-day funding rate.
    """
    try:
        # Map symbol to funding file path
        symbol_map = {
            'BTCUSDT': 'data/processed/funding/BTCUSDT.parquet',
            'ETHUSDT': 'data/processed/funding/ETHUSDT.parquet',
            'SOLUSDT': 'data/processed/funding/SOLUSDT.parquet'
        }
        
        if symbol not in symbol_map:
            return None
        
        df_funding = pd.read_parquet(symbol_map[symbol])
        if 'funding_rate' not in df_funding.columns:
            return None
        
        # Calculate 30-period rolling z-score
        funding = df_funding['funding_rate'].values
        n = len(funding)
        
        zscore = np.zeros(n)
        zscore[:] = np.nan
        
        for i in range(30, n):
            window = funding[i-30:i]
            mean_f = np.mean(window)
            std_f = np.std(window)
            if std_f > 1e-10:
                zscore[i] = (funding[i] - mean_f) / std_f
        
        return zscore
    except Exception:
        return None

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Get symbol from prices DataFrame (if available)
    symbol = prices.get('symbol', ['BTCUSDT'])[0] if 'symbol' in prices.columns else 'BTCUSDT'
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for ultra-long-term bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Load funding rate z-score (for BTC/ETH contrarian edge)
    funding_zscore = load_funding_data(symbol)
    if funding_zscore is None:
        funding_zscore = np.zeros(n)
        funding_zscore[:] = np.nan
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    hma_12h_fast = calculate_hma(close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # ATR ratio for volatility filter
    atr_short = calculate_atr(high, low, close, period=7)
    atr_long = calculate_atr(high, low, close, period=30)
    atr_ratio = atr_short / atr_long
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25  # 25% base position size
    SIZE_STRONG = 0.30  # 30% for strong signals
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h[i]) or np.isnan(chop[i]) or np.isnan(rsi[i]):
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
        
        # === HTF BIAS (1d HMA50) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === ULTRA HTF BIAS (1w HMA21) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else htf_1d_bull
        htf_1w_bear = close[i] < hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else htf_1d_bear
        
        # === REGIME FILTER (Choppiness Index) ===
        not_too_choppy = chop[i] < 62.0  # Avoid dead chop
        
        # === 12h HMA TREND ===
        hma_bull = close[i] > hma_12h[i]
        hma_bear = close[i] < hma_12h[i]
        
        # === HMA CROSSOVER (fast vs slow) ===
        hma_cross_bull = hma_12h_fast[i] > hma_12h[i] if not np.isnan(hma_12h_fast[i]) else False
        hma_cross_bear = hma_12h_fast[i] < hma_12h[i] if not np.isnan(hma_12h_fast[i]) else False
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI PULLBACK VALUES ===
        rsi_pullback_long = 35.0 < rsi[i] < 55.0  # Pullback in uptrend
        rsi_pullback_short = 45.0 < rsi[i] < 65.0  # Pullback in downtrend
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === VOLATILITY FILTER ===
        vol_ok = not np.isnan(atr_ratio[i]) and 0.5 < atr_ratio[i] < 3.0
        
        # === FUNDING RATE CONTRARIAN (BTC/ETH edge) ===
        funding_extreme_long = not np.isnan(funding_zscore[i]) and funding_zscore[i] < -1.5
        funding_extreme_short = not np.isnan(funding_zscore[i]) and funding_zscore[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRY: Trend + Pullback + HTF confirmation
        if hma_bull and htf_1d_bull and not_too_choppy and vol_ok:
            # Primary: RSI pullback in uptrend
            if rsi_pullback_long or rsi_oversold:
                desired_signal = SIZE_BASE
            
            # Strong: Add HMA crossover confirmation
            if (rsi_pullback_long or rsi_oversold) and hma_cross_bull:
                desired_signal = SIZE_STRONG
            
            # Bonus: Funding rate extreme (contrarian long)
            if funding_extreme_long and above_sma200:
                desired_signal = max(desired_signal, SIZE_STRONG)
        
        # SHORT ENTRY: Trend + Pullback + HTF confirmation
        elif hma_bear and htf_1d_bear and not_too_choppy and vol_ok:
            # Primary: RSI pullback in downtrend
            if rsi_pullback_short or rsi_overbought:
                desired_signal = -SIZE_BASE
            
            # Strong: Add HMA crossover confirmation
            if (rsi_pullback_short or rsi_overbought) and hma_cross_bear:
                desired_signal = -SIZE_STRONG
            
            # Bonus: Funding rate extreme (contrarian short)
            if funding_extreme_short and below_sma200:
                desired_signal = min(desired_signal, -SIZE_STRONG)
        
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
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
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