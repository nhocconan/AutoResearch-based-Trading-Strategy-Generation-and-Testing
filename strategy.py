#!/usr/bin/env python3
"""
Experiment #465: 15m Primary + 4h/1d HTF — Session-Filtered Pullback Strategy

Hypothesis: 15m timeframe is UNDEREXPLORED (0 successful experiments). Key insight:
- 15m generates too many signals → fee drag kills performance
- Solution: Use 4h/1d for TREND DIRECTION, 15m only for ENTRY TIMING
- Add SESSION FILTER (00-12 UTC) to trade only during high-liquidity periods
- Use RSI(7) pullback into value zone + volume confirmation for precision
- Target: 50-80 trades/year (strict entry filters to avoid fee drag)

Entry Logic:
- Long: 4h HMA bull + 1d ADX > 15 (some trend) + 15m RSI(7) < 35 (pullback) + volume > avg
- Short: 4h HMA bear + 1d ADX > 15 + 15m RSI(7) > 65 (pullback) + volume > avg
- Session: Only enter 00-12 UTC (London/NY overlap = best crypto liquidity)
- Breakout: 15m close > 4h Donchian upper (with HTF confirmation)

Risk Management:
- Position size: 0.15-0.25 (smaller for 15m frequency)
- Stoploss: 2.5x ATR from entry
- Take profit: Reduce to half at 2R, trail rest

Target: Sharpe>0.45, DD>-35%, trades>=60 train, trades>=10 test
Timeframe: 15m (FIRST 15m strategy with proper HTF + session filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_session_htf_rsi_pullback_v1"
timeframe = "15m"
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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_fisher_transform(high, low, close, period=9):
    """Ehlers Fisher Transform - catches reversals"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2.0
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        range_val = highest - lowest
        if range_val < 1e-10:
            continue
        
        x1 = (hl2 - lowest) / range_val
        x1 = np.clip(x1, 0.001, 0.999)
        
        fisher[i] = 0.5 * np.log((1.0 + x1) / (1.0 - x1))
        
        if i > 0 and not np.isnan(fisher[i-1]):
            fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    adx_1d_raw = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_raw)
    
    donchian_4h_upper_raw, donchian_4h_lower_raw = calculate_donchian(df_4h['high'].values, df_4h['low'].values, period=20)
    donchian_4h_upper = align_htf_to_ltf(prices, df_4h, donchian_4h_upper_raw)
    donchian_4h_lower = align_htf_to_ltf(prices, df_4h, donchian_4h_lower_raw)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=21)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Volume MA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    tp_hit = False
    
    for i in range(500, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m[i]) or np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC only) ===
        # Convert open_time (ms) to hour
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        in_session = (hour_utc >= 0 and hour_utc < 12)
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 1d REGIME FILTER (ADX > 15 = some trend) ===
        daily_adx = adx_1d_aligned[i]
        has_trend = not np.isnan(daily_adx) and daily_adx > 15.0
        
        # === 15m MOMENTUM (RSI pullback) ===
        rsi_oversold = rsi_7[i] < 35.0
        rsi_overbought = rsi_7[i] > 65.0
        
        # RSI recovery confirmation
        rsi_rising = rsi_7[i] > rsi_7[i-1] if i > 0 and not np.isnan(rsi_7[i-1]) else False
        rsi_falling = rsi_7[i] < rsi_7[i-1] if i > 0 and not np.isnan(rsi_7[i-1]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > vol_sma[i] * 1.2 if not np.isnan(vol_sma[i]) else False
        
        # === FISHER TRANSFORM REVERSAL ===
        fisher_long = False
        fisher_short = False
        if not np.isnan(fisher[i]) and not np.isnan(fisher_signal[i]):
            fisher_long = fisher_signal[i] < -1.5 and fisher[i] > fisher_signal[i]
            fisher_short = fisher_signal[i] > 1.5 and fisher[i] < fisher_signal[i]
        
        # === PRICE POSITION ===
        above_sma50 = close[i] > sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma50 = close[i] < sma_50[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === DONCHIAN BREAKOUT (4h HTF) ===
        donchian_breakout_long = False
        donchian_breakdown_short = False
        if not np.isnan(donchian_4h_upper[i]) and not np.isnan(donchian_4h_lower[i]):
            donchian_breakout_long = close[i] > donchian_4h_upper[i]
            donchian_breakdown_short = close[i] < donchian_4h_lower[i]
        
        # === ENTRY LOGIC (3+ confluence required) ===
        desired_signal = 0.0
        
        # LONG ENTRY: HTF bull + trend regime + RSI pullback + session
        if htf_bull and has_trend and in_session:
            confluence_count = 0
            
            # RSI pullback into value
            if rsi_oversold and rsi_rising:
                confluence_count += 1
            
            # Fisher reversal
            if fisher_long:
                confluence_count += 1
            
            # Price above SMA200 (long-term bull)
            if above_sma200:
                confluence_count += 1
            
            # Volume confirmation
            if vol_confirmed:
                confluence_count += 1
            
            # Donchian breakout
            if donchian_breakout_long:
                confluence_count += 1
            
            # Need 2+ confluence for entry
            if confluence_count >= 2:
                desired_signal = SIZE_STRONG if confluence_count >= 3 else SIZE_BASE
        
        # SHORT ENTRY: HTF bear + trend regime + RSI pullback + session
        elif htf_bear and has_trend and in_session:
            confluence_count = 0
            
            # RSI pullback into value
            if rsi_overbought and rsi_falling:
                confluence_count += 1
            
            # Fisher reversal
            if fisher_short:
                confluence_count += 1
            
            # Price below SMA200 (long-term bear)
            if below_sma200:
                confluence_count += 1
            
            # Volume confirmation
            if vol_confirmed:
                confluence_count += 1
            
            # Donchian breakdown
            if donchian_breakdown_short:
                confluence_count += 1
            
            # Need 2+ confluence for entry
            if confluence_count >= 2:
                desired_signal = -SIZE_STRONG if confluence_count >= 3 else -SIZE_BASE
        
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
        
        # === TAKE PROFIT (reduce to half at 2R) ===
        if in_position and not tp_hit:
            if position_side > 0:
                tp_level = entry_price + 2.0 * entry_atr
                if high[i] >= tp_level:
                    tp_hit = True
                    if desired_signal == 0.0:
                        desired_signal = position_side * SIZE_BASE / 2.0  # Reduce to half
            elif position_side < 0:
                tp_level = entry_price - 2.0 * entry_atr
                if low[i] <= tp_level:
                    tp_hit = True
                    if desired_signal == 0.0:
                        desired_signal = position_side * SIZE_BASE / 2.0  # Reduce to half
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif desired_signal > 0:
            final_signal = SIZE_BASE / 2.0
        elif desired_signal < 0:
            final_signal = -SIZE_BASE / 2.0
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
                tp_hit = False
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
                tp_hit = False
        
        signals[i] = final_signal
    
    return signals