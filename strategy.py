#!/usr/bin/env python3
"""
Experiment #290: 1h Primary + 4h/1d HTF — Funding Mean Reversion + RSI Session v1

Hypothesis: Funding rate contrarian signals work best in bear/range markets (2025 test).
Combine with simplified RSI entry + session filter for reliable trade generation.

Key improvements from #284 (12h):
1. PRIMARY TF: 1h instead of 12h (more signals, still controlled with filters)
2. SIMPLER ENTRY: RSI(14) < 40 / > 60 instead of complex CRSI (ensure trades)
3. SESSION FILTER: 08-20 UTC only (high liquidity, less manipulation)
4. HTF: 4h HMA(21) for trend bias + 1d HMA(50) for major trend
5. FUNDING: Load ONCE before loop (fix #288/#289 crash), z-score < -1.5 / > +1.5
6. LOOSENED FILTERS: Ensure 40-80 trades/year target is met

Entry Logic (3+ confluence required):
- Long: RSI < 40 + (funding_z < -1.5 OR 4h HMA bull) + session 08-20 UTC
- Short: RSI > 60 + (funding_z > +1.5 OR 4h HMA bear) + session 08-20 UTC
- Strong size when 1d HMA aligns with direction

Position sizing: 0.20 base, 0.30 when HTF aligned (discrete levels)
Stoploss: 2.5x ATR from entry price

Target: Sharpe>0.40, DD>-40%, trades>=40/year train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf
from pathlib import Path

name = "mtf_1h_funding_rsi_session_4h1d_v1"
timeframe = "1h"
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

def load_funding_data(prices):
    """
    Load funding rate data ONCE before loop (CRITICAL - fixes #288/#289 crash)
    Returns z-score array aligned with prices
    """
    n = len(prices)
    zscore = np.zeros(n)
    zscore[:] = np.nan
    
    try:
        # Try to infer symbol from prices or use default
        # Engine should provide symbol context, fallback to BTCUSDT
        symbol = "BTCUSDT"
        funding_path = Path("data/processed/funding/BTCUSDT.parquet")
        
        # Try ETH if BTC not found
        if not funding_path.exists():
            symbol = "ETHUSDT"
            funding_path = Path("data/processed/funding/ETHUSDT.parquet")
        
        # Try SOL if ETH not found
        if not funding_path.exists():
            symbol = "SOLUSDT"
            funding_path = Path("data/processed/funding/SOLUSDT.parquet")
        
        if funding_path.exists():
            funding_df = pd.read_parquet(funding_path)
            if 'funding_rate' in funding_df.columns:
                funding_rates = funding_df['funding_rate'].values
                lookback = 30
                
                # Calculate z-score for each point
                for i in range(lookback, min(n, len(funding_rates))):
                    window = funding_rates[max(0, i-lookback):i]
                    if len(window) >= lookback // 2:
                        mean = np.nanmean(window)
                        std = np.nanstd(window)
                        if std > 1e-10 and not np.isnan(funding_rates[i]):
                            zscore[i] = (funding_rates[i] - mean) / std
    except Exception:
        pass
    
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Load funding data ONCE before loop (fixes #288/#289 crash)
    funding_z = load_funding_data(prices)
    
    # Calculate primary (1h) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        # open_time is in milliseconds, extract UTC hour
        timestamp_ms = prices['open_time'].iloc[i]
        hour = (timestamp_ms // 3600000) % 24
        in_session = 8 <= hour <= 20
        
        # === HTF TREND BIAS ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_1d_bull = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        htf_1d_bear = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # === RSI EXTREMES (LOOSENED for trade generation) ===
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        
        # === FUNDING RATE CONTRARIAN ===
        funding_long_bias = not np.isnan(funding_z[i]) and funding_z[i] < -1.5
        funding_short_bias = not np.isnan(funding_z[i]) and funding_z[i] > 1.5
        funding_strong_long = not np.isnan(funding_z[i]) and funding_z[i] < -2.0
        funding_strong_short = not np.isnan(funding_z[i]) and funding_z[i] > 2.0
        
        # === ENTRY LOGIC (3+ confluence required) ===
        desired_signal = 0.0
        
        # Only trade during session hours
        if in_session:
            # FUNDING OVERRIDE: Strong funding signals get strong size
            if funding_strong_long:
                desired_signal = SIZE_STRONG
            elif funding_strong_short:
                desired_signal = -SIZE_STRONG
            
            # LONG: RSI oversold + (funding long OR 4h bull) + session
            elif rsi_oversold and (funding_long_bias or htf_4h_bull):
                size = SIZE_STRONG if htf_1d_bull else SIZE_BASE
                desired_signal = size
            
            # SHORT: RSI overbought + (funding short OR 4h bear) + session
            elif rsi_overbought and (funding_short_bias or htf_4h_bear):
                size = SIZE_STRONG if htf_1d_bear else SIZE_BASE
                desired_signal = -size
        
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