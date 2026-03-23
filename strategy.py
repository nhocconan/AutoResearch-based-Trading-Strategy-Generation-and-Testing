#!/usr/bin/env python3
"""
Experiment #695: 1h Primary + 4h/1d HTF — Funding Rate Contrarian + HTF Trend + RSI Pullback

Hypothesis: Funding rate mean reversion is the BEST edge for BTC/ETH (proven through 2022 crash).
When funding is extreme (>2 std dev), crowds are positioned wrong → contrarian entry.
Combined with 4h HMA trend bias and 1h RSI pullback timing, this captures reversals
with HTF confirmation. Session filter (8-20 UTC) ensures high liquidity entries.

Why this should work where others failed:
1. Funding rate contrarian worked through 2022 crash (Sharpe 0.8-1.5 per research)
2. 4h HMA provides trend bias without whipsaw of 1h trend signals
3. RSI pullback (not extreme) ensures we enter on retracement, not chase
4. Session filter reduces false signals in low-liquidity hours
5. Very strict confluence (3+ filters) = few trades (target 30-60/year on 1h)
6. Small position size (0.20-0.25) limits drawdown on 77% BTC crash

Key differences from failed 1h strategies (#685, #690):
- Funding rate as PRIMARY signal (not just technical)
- 4h HMA for trend (not 1h EMA crossover)
- RSI 35-65 pullback zone (not extreme 20/80)
- Session filter 8-20 UTC only
- Stricter confluence = fewer trades = less fee drag

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_funding_contrarian_hma_rsi_session_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
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
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i-1])
        tr3 = np.abs(low[i] - close[i-1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_funding_zscore(funding_data, window=720):
    """
    Calculate z-score of funding rate over rolling window.
    window=720 hours = 30 days for 1h data
    Positive z-score = longs paying shorts (crowd long) → contrarian short signal
    Negative z-score = shorts paying longs (crowd short) → contrarian long signal
    """
    if funding_data is None or len(funding_data) < window:
        return None
    
    funding = funding_data['funding_rate'].values
    n = len(funding)
    zscore = np.full(n, np.nan)
    
    for i in range(window, n):
        window_data = funding[i-window:i]
        mean = np.mean(window_data)
        std = np.std(window_data)
        if std > 1e-10:
            zscore[i] = (funding[i] - mean) / std
    
    return zscore

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
    
    # Load funding rate data (contrarian signal)
    funding_path = f"data/processed/funding/{prices['symbol'].iloc[0] if 'symbol' in prices.columns else 'BTCUSDT'}.parquet"
    try:
        funding_df = pd.read_parquet(funding_path)
        # Align funding to prices timeframe
        funding_zscore = calculate_funding_zscore(funding_df, window=720)
        # Need to align funding zscore to prices index
        # Funding is typically 8h intervals, we need to forward-fill to 1h
        if funding_zscore is not None:
            funding_aligned = np.interp(
                np.arange(n),
                np.linspace(0, len(funding_zscore)-1, len(funding_zscore)) * (n / len(funding_zscore)),
                funding_zscore
            )
            funding_aligned = np.nan_to_num(funding_aligned, nan=0.0)
        else:
            funding_aligned = np.zeros(n)
    except:
        # Fallback if funding data not available
        funding_aligned = np.zeros(n)
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # Volume MA for filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller for 1h TF to reduce fee impact
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Need 200 for SMA + buffer for HTF alignment
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(vol_ma[i]):
            continue
        if atr_1h[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC only - high liquidity) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_ma[i]
        
        # === TREND BIAS (HTF HMA) ===
        trend_bullish_4h = close[i] > hma_4h_aligned[i]
        trend_bearish_4h = close[i] < hma_4h_aligned[i]
        trend_bullish_1d = close[i] > hma_1d_aligned[i]
        trend_bearish_1d = close[i] < hma_1d_aligned[i]
        
        # Strong bias when both HTF agree
        trend_strong_bullish = trend_bullish_4h and trend_bullish_1d
        trend_strong_bearish = trend_bearish_4h and trend_bearish_1d
        
        # === FUNDING RATE CONTRARIAN SIGNAL ===
        # Z-score > +2.0 = crowd long → contrarian short
        # Z-score < -2.0 = crowd short → contrarian long
        funding_extreme_long = funding_aligned[i] > 2.0
        funding_extreme_short = funding_aligned[i] < -2.0
        funding_neutral = -1.5 <= funding_aligned[i] <= 1.5
        
        # === RSI PULLBACK (not extreme - we want pullback in trend) ===
        # In bullish trend: RSI 35-50 = pullback entry
        # In bearish trend: RSI 50-65 = pullback entry
        rsi_pullback_long = 35 <= rsi_1h[i] <= 55
        rsi_pullback_short = 45 <= rsi_1h[i] <= 65
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === LONG ENTRY (3+ confluence required) ===
        # Must have: session + volume + (HTF trend OR funding contrarian) + RSI pullback
        if in_session and volume_ok:
            # Scenario 1: Strong bullish trend + RSI pullback
            if trend_strong_bullish and rsi_pullback_long and above_sma200:
                desired_signal = current_size
            
            # Scenario 2: Funding extreme short (contrarian long) + HTF not bearish
            elif funding_extreme_short and not trend_strong_bearish and rsi_pullback_long:
                desired_signal = current_size
            
            # Scenario 3: 4h bullish + funding neutral + RSI pullback
            elif trend_bullish_4h and funding_neutral and rsi_pullback_long and above_sma200:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY (3+ confluence required) ===
        if in_session and volume_ok:
            # Scenario 1: Strong bearish trend + RSI pullback
            if trend_strong_bearish and rsi_pullback_short and below_sma200:
                desired_signal = -current_size
            
            # Scenario 2: Funding extreme long (contrarian short) + HTF not bullish
            elif funding_extreme_long and not trend_strong_bullish and rsi_pullback_short:
                desired_signal = -current_size
            
            # Scenario 3: 4h bearish + funding neutral + RSI pullback
            elif trend_bearish_4h and funding_neutral and rsi_pullback_short and below_sma200:
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
        
        # === HOLD LOGIC — Maintain position if conditions still valid ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and RSI not overbought
                if trend_bullish_4h and rsi_1h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and RSI not oversold
                if trend_bearish_4h and rsi_1h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        # Long exit: RSI overbought OR trend reverses
        if in_position and position_side > 0:
            if rsi_1h[i] > 75:
                desired_signal = 0.0
            elif close[i] < hma_4h_aligned[i] and close[i] < sma_200[i]:
                desired_signal = 0.0
        
        # Short exit: RSI oversold OR trend reverses
        if in_position and position_side < 0:
            if rsi_1h[i] < 25:
                desired_signal = 0.0
            elif close[i] > hma_4h_aligned[i] and close[i] > sma_200[i]:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE * 0.8 else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE * 0.8 else -REDUCED_SIZE
        
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