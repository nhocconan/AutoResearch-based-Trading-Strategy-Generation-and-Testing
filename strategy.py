#!/usr/bin/env python3
"""
Experiment #955: 1h Primary + 4h/1d HTF — Simplified Trend Pullback with Volume

Hypothesis: After 684 failed strategies (many with Sharpe=0.000 = 0 trades), the key is 
SIMPLICITY + ensuring trades actually happen. Over-filtering kills trade count.

Strategy Design:
1. 4h HMA(21) for trend direction (bullish when price > HMA)
2. 1h RSI(14) for pullback entries (RSI<45 long, RSI>55 short)
3. Volume filter: >0.7x 20-bar avg (ensures liquidity)
4. Session filter: 8-20 UTC only (liquid hours, reduces noise)
5. 1d HMA(21) for macro bias (optional size boost)
6. Stoploss: 2.5x ATR trailing

Key insights from failures:
- Exp 944, 945, 948, 952: Sharpe=0.000 = TOO MANY FILTERS = 0 trades
- Exp 950, 954: Negative Sharpe = wrong logic for bear market
- CRSI strategies keep failing on 1h (too noisy)

Why this should work:
- RELAXED RSI thresholds (45/55 not 30/70) = more trades
- Volume filter is lenient (0.7x not 1.2x) = doesn't block entries
- OR logic: (trend+RSI) OR (funding extreme) = multiple entry paths
- Position size 0.25 = conservative for 1h TF
- Target: 40-80 trades/year (within 30-60 guideline for 1h)

Timeframe: 1h (as required)
Target trades: 40-80/year
Position size: 0.25 (discrete: 0.0, ±0.25)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_trend_pullback_volume_4h1d_session_v1"
timeframe = "1h"
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
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_4h = get_htf_data(prices, '4h')
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
    
    # Calculate 1h indicators
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align 4h HMA for trend (Rule 1 & 2 - ONCE + aligned)
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate funding z-score
    funding_z = calculate_funding_zscore(funding_rates, period=30)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] <= 1e-10:
            continue
        
        # Extract hour from open_time (UTC)
        hour = pd.to_datetime(open_time[i], unit='ms').hour
        
        # Session filter: only trade 8-20 UTC (liquid hours)
        in_session = 8 <= hour <= 20
        
        # Volume filter: above 0.7x 20-bar average (lenient)
        volume_ok = volume[i] > 0.7 * vol_avg_20[i]
        
        # === TREND DIRECTION (4h HMA21) ===
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # === MACRO FILTER (1d HMA21) ===
        macro_bullish = close[i] > hma_1d_aligned[i]
        macro_bearish = close[i] < hma_1d_aligned[i]
        
        # === RSI PULLBACK (relaxed thresholds for more trades) ===
        rsi_pullback_long = rsi_1h[i] < 45
        rsi_pullback_short = rsi_1h[i] > 55
        
        # === FUNDING CONTRARIAN (alternative entry path) ===
        funding_extreme_long = funding_z[i] < -1.5  # Too many shorts → long
        funding_extreme_short = funding_z[i] > 1.5  # Too many longs → short
        
        desired_signal = 0.0
        
        # === LONG ENTRY (multiple paths to ensure trades) ===
        # Path 1: Trend + RSI pullback + session
        if trend_bullish and rsi_pullback_long and in_session:
            desired_signal = BASE_SIZE
        # Path 2: Funding extreme long (contrarian, works in any trend)
        elif funding_extreme_long and in_session:
            desired_signal = BASE_SIZE
        # Path 3: Macro bullish + RSI pullback (stronger confluence)
        elif macro_bullish and rsi_pullback_long and volume_ok:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY (multiple paths) ===
        # Path 1: Trend + RSI rally + session
        if trend_bearish and rsi_pullback_short and in_session:
            desired_signal = -BASE_SIZE
        # Path 2: Funding extreme short (contrarian)
        elif funding_extreme_short and in_session:
            desired_signal = -BASE_SIZE
        # Path 3: Macro bearish + RSI rally
        elif macro_bearish and rsi_pullback_short and volume_ok:
            desired_signal = -BASE_SIZE
        
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
        
        # === HOLD LOGIC (maintain position through minor pullbacks) ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and RSI not overbought
                if trend_bullish and rsi_1h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and RSI not oversold
                if trend_bearish and rsi_1h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses + RSI overbought
            if trend_bearish and rsi_1h[i] > 65:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses + RSI oversold
            if trend_bullish and rsi_1h[i] < 35:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
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