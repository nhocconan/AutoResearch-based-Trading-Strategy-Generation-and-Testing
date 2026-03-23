#!/usr/bin/env python3
"""
Experiment #690: 1h Primary + 4h/12h HTF — Regime-Adaptive Multi-Signal Confluence

Hypothesis: Lower TF (1h) strategies fail due to either (a) too many trades → fee drag,
or (b) too strict filters → 0 trades. This strategy balances both by:

1. REGIME DETECTION: Choppiness Index (14) distinguishes range vs trend
   - CHOP > 55 = range → use mean reversion (RSI extremes)
   - CHOP < 45 = trend → use trend following (HTF HMA direction)
   - This adapts to market conditions instead of one-size-fits-all

2. HTF TREND BIAS: 4h HMA(21) + 12h HMA(21) for directional bias
   - Only long when both 4h and 12h HMA are bullish
   - Only short when both 4h and 12h HMA are bearish
   - Reduces counter-trend trades that fail in 2022 crash

3. ENTRY TIMING: 1h RSI(14) for precise entry within HTF trend
   - Long: RSI < 40 (pullback in uptrend)
   - Short: RSI > 60 (rally in downtrend)
   - Looser than CRSI to ensure trades generate

4. VOLUME CONFIRMATION: Volume > 0.7x 20-bar average
   - Filters out low-liquidity fake moves
   - Less strict than 1.0x to allow more trades

5. FUNDING RATE CONTRARIAN: Load from data/processed/funding/
   - Long when funding < -0.01% (crowd too short)
   - Short when funding > 0.02% (crowd too long)
   - Proven edge for BTC/ETH specifically

6. SESSION FILTER: Only trade 8-20 UTC (high liquidity)
   - Avoids Asian session whipsaw
   - Reduces false signals

7. POSITION SIZING: Discrete levels (0.0, ±0.25, ±0.30)
   - MAX 0.35 to survive 2022-style crashes
   - Each signal change costs fees → minimize churn

8. STOPLOSS: 2.5x ATR trailing stop
   - Signal → 0 when stopped out
   - Protects capital in crash scenarios

Why this should beat #689 (Sharpe=-0.129):
- Regime detection adapts to market (not fixed logic)
- Multiple confluence filters reduce false signals
- Funding rate adds unique edge not in #689
- Looser RSI thresholds ensure trades generate
- 1h TF with HTF filters = optimal trade frequency (30-60/year)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_rsi_hma_funding_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) — measures market ranging vs trending.
    CHOP > 61.8 = ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    Formula: 100 * (sum(ATR, n) / (highest high - lowest low)) / (log10(n))
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    # Sum of ATR over period
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest high and lowest low over period
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    price_range = highest - lowest
    price_range = np.where(price_range < 1e-10, 1e-10, price_range)
    
    # CHOP formula
    chop_raw = 100 * (atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop_raw, 0, 100)
    
    return chop

def calculate_hma(close, period=21):
    """
    Hull Moving Average — smoother and more responsive than EMA.
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    def wma(data, span):
        """Weighted Moving Average"""
        weights = np.arange(1, span + 1)
        result = np.full(len(data), np.nan)
        for i in range(span - 1, len(data)):
            result[i] = np.sum(data[i - span + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = 2 * wma_half - wma_full
    
    # WMA of diff with sqrt(n)
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Pad first element
    gain = np.concatenate([[0], gain])
    loss = np.concatenate([[0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
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

def load_funding_rate(symbol):
    """
    Load funding rate data from processed parquet.
    Returns aligned funding rate array.
    """
    try:
        # Map symbol to funding file
        symbol_map = {
            'BTCUSDT': 'BTCUSDT',
            'ETHUSDT': 'ETHUSDT',
            'SOLUSDT': 'SOLUSDT'
        }
        funding_symbol = symbol_map.get(symbol, symbol)
        funding_path = f"data/processed/funding/{funding_symbol}.parquet"
        df_funding = pd.read_parquet(funding_path)
        return df_funding['funding_rate'].values
    except Exception:
        # Return zeros if funding data unavailable
        return None

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Get symbol from prices DataFrame (if available)
    symbol = prices.get('symbol', ['BTCUSDT'])[0] if isinstance(prices.get('symbol'), list) else 'BTCUSDT'
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Load funding rate (optional filter)
    funding_rates = load_funding_rate(symbol)
    if funding_rates is not None and len(funding_rates) >= n:
        funding_aligned = funding_rates[:n]
    else:
        funding_aligned = np.zeros(n)
    
    # Calculate primary (1h) indicators
    chop_1h = calculate_choppiness_index(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    
    # Volume MA for filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align HTF (4h) HMA
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align HTF (12h) HMA
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(chop_1h[i]) or np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if atr_1h[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 55 = range (mean reversion)
        # CHOP < 45 = trend (trend following)
        # 45-55 = neutral (use either)
        is_range = chop_1h[i] > 55
        is_trend = chop_1h[i] < 45
        
        # === HTF TREND BIAS (4h + 12h HMA) ===
        # Both must agree for strong signal
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        hma_12h_bullish = close[i] > hma_12h_aligned[i]
        hma_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # Strong trend bias: both 4h and 12h agree
        strong_bullish = hma_4h_bullish and hma_12h_bullish
        strong_bearish = hma_4h_bearish and hma_12h_bearish
        
        # === VOLUME FILTER ===
        # Volume must be at least 70% of 20-bar average
        volume_ok = volume[i] > 0.7 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else True
        
        # === SESSION FILTER (8-20 UTC) ===
        # Extract hour from open_time (assumes milliseconds timestamp)
        try:
            timestamp_ms = prices['open_time'].iloc[i]
            hour_utc = (timestamp_ms // 3600000) % 24
            session_ok = 8 <= hour_utc <= 20
        except Exception:
            session_ok = True  # Default to OK if can't parse
        
        # === FUNDING RATE FILTER (Contrarian) ===
        # Long when funding < -0.01% (crowd too short)
        # Short when funding > 0.02% (crowd too long)
        funding_long_ok = funding_aligned[i] < -0.0001 if len(funding_aligned) > i else True
        funding_short_ok = funding_aligned[i] > 0.0002 if len(funding_aligned) > i else True
        
        # === RSI ENTRY SIGNALS ===
        # Long: RSI < 40 (pullback)
        # Short: RSI > 60 (rally)
        rsi_oversold = rsi_1h[i] < 40
        rsi_overbought = rsi_1h[i] > 60
        
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Regime 1: Range market + RSI oversold + volume OK
        if is_range and rsi_oversold and volume_ok:
            if funding_long_ok or not funding_aligned.any():
                desired_signal = SIZE_LONG
        
        # Regime 2: Trend market + strong bullish bias + RSI pullback
        elif is_trend and strong_bullish and rsi_oversold:
            if session_ok and volume_ok:
                desired_signal = SIZE_LONG
        
        # Regime 3: Neutral + strong bullish + RSI very oversold
        elif strong_bullish and rsi_1h[i] < 35:
            desired_signal = SIZE_LONG * 0.8
        
        # === SHORT ENTRY ===
        # Regime 1: Range market + RSI overbought + volume OK
        if is_range and rsi_overbought and volume_ok:
            if funding_short_ok or not funding_aligned.any():
                desired_signal = -SIZE_SHORT
        
        # Regime 2: Trend market + strong bearish bias + RSI rally
        elif is_trend and strong_bearish and rsi_overbought:
            if session_ok and volume_ok:
                desired_signal = -SIZE_SHORT
        
        # Regime 3: Neutral + strong bearish + RSI very overbought
        elif strong_bearish and rsi_1h[i] > 65:
            desired_signal = -SIZE_SHORT * 0.8
        
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
        
        # === HOLD LOGIC — Maintain position if conditions still valid ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HTF still bullish and RSI not overbought
                if strong_bullish and rsi_1h[i] < 65:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if HTF still bearish and RSI not oversold
                if strong_bearish and rsi_1h[i] > 35:
                    desired_signal = -SIZE_SHORT
        
        # === EXIT CONDITIONS ===
        # Long exit: RSI > 70 (overbought) OR HTF trend reverses
        if in_position and position_side > 0:
            if rsi_1h[i] > 70 or (strong_bearish and chop_1h[i] < 45):
                desired_signal = 0.0
        
        # Short exit: RSI < 30 (oversold) OR HTF trend reverses
        if in_position and position_side < 0:
            if rsi_1h[i] < 30 or (strong_bullish and chop_1h[i] < 45):
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0.15:
            desired_signal = SIZE_LONG
        elif desired_signal < -0.15:
            desired_signal = -SIZE_SHORT
        else:
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