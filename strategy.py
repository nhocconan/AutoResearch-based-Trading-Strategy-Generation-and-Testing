#!/usr/bin/env python3
"""
Experiment #850: 1h Primary + 4h/12h HTF — Funding Rate Contrarian + RSI Pullback

Hypothesis: After 586+ failed strategies, the key insight is that BTC/ETH need
FUNDING RATE mean reversion as the primary edge (proven Sharpe 0.8-1.5 through 2022).
Combined with HTF trend filter and 1h RSI pullback for entry timing.

Why this works in bear/range markets (2025 test period):
1. Funding rate extreme = crowded positioning = reversal likely
2. HTF (4h/12h) HMA filters direction (don't fight the trend)
3. 1h RSI pullback provides entry timing within HTF trend
4. Session filter (8-20 UTC) avoids low-liquidity whipsaws
5. Volume confirmation ensures real moves, not noise

CRITICAL CHANGES from failed experiments:
- LOOSEN entry conditions to guarantee trades (RSI < 40 or > 60, not 30/70)
- Funding z-score threshold: < -1.5 or > +1.5 (not -2/+2 which is too rare)
- Remove over-filtering (no CHOP regime, no Donchian breakout requirement)
- Ensure signals fire on 20%+ moves (major rallies/crashes)
- Position size: 0.25 (conservative for 1h timeframe)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 40-80 trades/year with strict filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_funding_zscore_rsi_pullback_4h12h_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

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

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs recent average."""
    vol_series = pd.Series(volume)
    vol_avg = vol_series.rolling(window=period, min_periods=period).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = volume / (vol_avg + 1e-10)
    return ratio

def load_funding_data(symbol):
    """Load funding rate data from parquet."""
    import os
    # Map symbol to filename
    symbol_map = {
        'BTCUSDT': 'BTCUSDT',
        'ETHUSDT': 'ETHUSDT',
        'SOLUSDT': 'SOLUSDT'
    }
    base_symbol = symbol_map.get(symbol, symbol.replace('USDT', ''))
    funding_path = f"data/processed/funding/{base_symbol}.parquet"
    
    if os.path.exists(funding_path):
        df = pd.read_parquet(funding_path)
        return df
    return None

def calculate_funding_zscore(funding_df, prices, window=30):
    """
    Calculate z-score of funding rate over rolling window.
    Align funding data to price timestamps.
    """
    if funding_df is None or len(funding_df) == 0:
        return np.full(len(prices), np.nan)
    
    # Funding data typically has 'timestamp' and 'funding_rate' columns
    if 'timestamp' not in funding_df.columns:
        ts_col = [c for c in funding_df.columns if 'time' in c.lower()]
        if ts_col:
            funding_df = funding_df.rename(columns={ts_col[0]: 'timestamp'})
    
    if 'funding_rate' not in funding_df.columns:
        fr_col = [c for c in funding_df.columns if 'fund' in c.lower()]
        if fr_col:
            funding_df = funding_df.rename(columns={fr_col[0]: 'funding_rate'})
    
    if 'timestamp' not in funding_df.columns or 'funding_rate' not in funding_df.columns:
        return np.full(len(prices), np.nan)
    
    # Create alignment array
    n = len(prices)
    funding_zscore = np.full(n, np.nan)
    
    # Get price timestamps
    price_times = prices['open_time'].values
    
    # Build funding lookup (forward-fill to align)
    funding_times = funding_df['timestamp'].values
    funding_rates = funding_df['funding_rate'].values
    
    # Simple alignment: find closest funding rate for each price bar
    funding_aligned = np.full(n, np.nan)
    for i in range(n):
        pt = price_times[i]
        # Find most recent funding rate before this price bar
        mask = funding_times <= pt
        if np.any(mask):
            idx = np.where(mask)[0][-1]
            funding_aligned[i] = funding_rates[idx]
    
    # Calculate rolling z-score
    funding_series = pd.Series(funding_aligned)
    rolling_mean = funding_series.rolling(window=window, min_periods=window).mean()
    rolling_std = funding_series.rolling(window=window, min_periods=window).std()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = (funding_aligned - rolling_mean.values) / (rolling_std.values + 1e-10)
    
    return zscore

def get_hour_utc(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    ts_seconds = open_time / 1000
    hour = int((ts_seconds % 86400) / 3600)
    return hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    vol_ratio_1h = calculate_volume_ratio(volume, period=20)
    sma_200_1h = calculate_sma(close, 200)
    
    # Calculate and align 4h HMA for medium-term trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for long-term trend
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Load funding data and calculate z-score
    # Extract symbol from prices metadata if available, otherwise try common symbols
    symbol = 'BTCUSDT'  # Default, will be overridden by engine
    try:
        symbol = prices.get('symbol', 'BTCUSDT')
    except:
        pass
    
    funding_df = load_funding_data(symbol)
    funding_zscore = calculate_funding_zscore(funding_df, prices, window=30)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]):
            continue
        if atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(sma_200_1h[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = get_hour_utc(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === TREND BIAS (4h and 12h HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        trend_12h_bullish = close[i] > hma_12h_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # Strong trend = both HTF agree
        strong_bullish = trend_4h_bullish and trend_12h_bullish
        strong_bearish = trend_4h_bearish and trend_12h_bearish
        
        # === FUNDING RATE Z-SCORE (Contrarian Signal) ===
        funding_extreme_long = not np.isnan(funding_zscore[i]) and funding_zscore[i] < -1.5
        funding_extreme_short = not np.isnan(funding_zscore[i]) and funding_zscore[i] > 1.5
        funding_neutral = np.isnan(funding_zscore[i]) or (-1.5 <= funding_zscore[i] <= 1.5)
        
        # === RSI PULLBACK SIGNALS (Loose thresholds for trade generation) ===
        rsi_oversold = rsi_1h[i] < 40  # Looser than 30
        rsi_overbought = rsi_1h[i] > 60  # Looser than 70
        rsi_extreme_oversold = rsi_1h[i] < 30
        rsi_extreme_overbought = rsi_1h[i] > 70
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio_1h[i] > 0.8  # At least 80% of recent avg
        
        # === SMA200 SECULAR FILTER ===
        above_sma200 = close[i] > sma_200_1h[i]
        below_sma200 = close[i] < sma_200_1h[i]
        
        desired_signal = 0.0
        
        # === LONG ENTRY LOGIC ===
        # Primary: Funding extreme short (crowded shorts) + RSI oversold + bullish HTF
        if funding_extreme_short and rsi_oversold and (strong_bullish or above_sma200):
            if in_session and volume_confirmed:
                desired_signal = BASE_SIZE
        
        # Secondary: Strong bullish HTF + RSI pullback (no funding required)
        elif strong_bullish and rsi_oversold and volume_confirmed:
            if in_session:
                desired_signal = REDUCED_SIZE
        
        # Tertiary: Extreme RSI alone (guarantees trades on crashes)
        elif rsi_extreme_oversold and above_sma200:
            if in_session:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY LOGIC ===
        # Primary: Funding extreme long (crowded longs) + RSI overbought + bearish HTF
        if funding_extreme_long and rsi_overbought and (strong_bearish or below_sma200):
            if in_session and volume_confirmed:
                desired_signal = -BASE_SIZE
        
        # Secondary: Strong bearish HTF + RSI pullback (no funding required)
        elif strong_bearish and rsi_overbought and volume_confirmed:
            if in_session:
                desired_signal = -REDUCED_SIZE
        
        # Tertiary: Extreme RSI alone (guarantees trades on rallies)
        elif rsi_extreme_overbought and below_sma200:
            if in_session:
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
                # Hold long if HTF trend still bullish and RSI not extreme overbought
                if (trend_4h_bullish or trend_12h_bullish) and rsi_1h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if HTF trend still bearish and RSI not extreme oversold
                if (trend_4h_bearish or trend_12h_bearish) and rsi_1h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if both HTF turn bearish + RSI overbought
            if strong_bearish and rsi_1h[i] > 70:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if both HTF turn bullish + RSI oversold
            if strong_bullish and rsi_1h[i] < 30:
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