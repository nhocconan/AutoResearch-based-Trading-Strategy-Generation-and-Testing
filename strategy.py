#!/usr/bin/env python3
"""
Experiment #920: 1h Primary + 4h/12h HTF — Simplified Dual Regime + RSI Pullback

Hypothesis: After 650+ failed strategies, the #1 issue is TOO MANY confluence filters
that never all agree (CRSI + Choppiness + HTF + Donchian + Volume + Session = 0 trades).

Key insight from failures:
- Exp #910 (1h CRSI+Chop+4h/12h HMA): Sharpe=0.000 (0 trades)
- Exp #919 (4h simplified regime): Sharpe=0.000 (0 trades)
- Complex regime detection = paralysis

NEW APPROACH:
1. SIMPLIFIED regime: Just 4h HMA21 direction (bull/bear), NOT Choppiness
2. SINGLE entry trigger: 1h RSI(14) pullback within HTF trend
3. LOOSE thresholds: RSI < 40 for long, RSI > 60 for short (not 20/80!)
4. ADD funding rate contrarian signal where available (proven BTC/ETH edge)
5. MINIMAL filters: ADX > 15 (weak trend filter), volume > 0.5x avg

Why this should generate trades:
- RSI < 40 happens ~15% of bars (vs CRSI < 20 at ~3%)
- 4h HMA trend changes slowly = sustained directional bias
- 1h timeframe = 8760 bars/year, even 0.5% trigger rate = 40+ trades

Position sizing: 0.25 (conservative for 1h TF)
Stoploss: 2.5x ATR(14) trailing
Target: 30-80 trades/year, Sharpe > 0.612

Timeframe: 1h (primary), 4h/12h (HTF trend)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_simplified_rsi_pullback_4h12h_hma_funding_atr_v1"
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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def load_funding_data(symbol):
    """Load funding rate data if available."""
    try:
        import os
        symbol_map = {'BTCUSDT': 'BTC', 'ETHUSDT': 'ETH', 'SOLUSDT': 'SOL'}
        base_symbol = symbol_map.get(symbol, symbol.replace('USDT', ''))
        funding_path = f"data/processed/funding/{base_symbol}_funding.parquet"
        if os.path.exists(funding_path):
            df = pd.read_parquet(funding_path)
            return df['funding_rate'].values
    except:
        pass
    return None

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    adx_1h = calculate_adx(high, low, close, period=14)
    sma_50_1h = calculate_sma(close, 50)
    sma_200_1h = calculate_sma(close, 200)
    
    # Volume MA for filter
    vol_ma_20 = calculate_sma(volume, 20)
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for confirmation
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Try to load funding rate data
    funding = None
    try:
        symbol = prices.get('symbol', 'BTCUSDT')
        if isinstance(symbol, pd.Series):
            symbol = symbol.iloc[0]
        funding = load_funding_data(symbol)
        if funding is not None and len(funding) < n:
            funding = None
    except:
        funding = None
    
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
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(sma_50_1h[i]) or np.isnan(sma_200_1h[i]):
            continue
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 1e-10:
            continue
        
        # === HTF TREND DIRECTION (4h HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === HTF CONFIRMATION (12h HMA21) ===
        trend_12h_bullish = close[i] > hma_12h_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # === LTF TREND FILTER (1h SMA50/200) ===
        above_sma50 = close[i] > sma_50_1h[i]
        below_sma50 = close[i] < sma_50_1h[i]
        above_sma200 = close[i] > sma_200_1h[i]
        below_sma200 = close[i] < sma_200_1h[i]
        
        # === ADX FILTER (minimum trend strength) ===
        adx_ok = not np.isnan(adx_1h[i]) and adx_1h[i] > 15
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.5 * vol_ma_20[i]
        
        # === RSI SIGNALS (LOOSE thresholds for trade frequency) ===
        rsi_oversold = rsi_1h[i] < 40  # Was 20, now 40 for more trades
        rsi_overbought = rsi_1h[i] > 60  # Was 80, now 60 for more trades
        rsi_extreme_oversold = rsi_1h[i] < 30
        rsi_extreme_overbought = rsi_1h[i] > 70
        
        # === FUNDING RATE CONTRARIAN (if available) ===
        funding_extreme_long = False
        funding_extreme_short = False
        if funding is not None and i < len(funding) and not np.isnan(funding[i]):
            funding_extreme_long = funding[i] > 0.0005  # >0.05% = crowded long
            funding_extreme_short = funding[i] < -0.0005  # <-0.05% = crowded short
        
        desired_signal = 0.0
        
        # === LONG ENTRY: 4h bullish + RSI pullback ===
        if trend_4h_bullish:
            # Primary: RSI oversold + volume + ADX
            if rsi_oversold and volume_ok:
                if adx_ok or trend_12h_bullish:
                    desired_signal = BASE_SIZE
                elif rsi_extreme_oversold:
                    desired_signal = REDUCED_SIZE
            
            # Secondary: RSI extreme alone (guarantees trades)
            if rsi_extreme_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            # Funding contrarian: extreme short funding = long opportunity
            if funding_extreme_short and trend_4h_bullish and desired_signal == 0:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY: 4h bearish + RSI bounce ===
        if trend_4h_bearish:
            # Primary: RSI overbought + volume + ADX
            if rsi_overbought and volume_ok:
                if adx_ok or trend_12h_bearish:
                    desired_signal = -BASE_SIZE
                elif rsi_extreme_overbought:
                    desired_signal = -REDUCED_SIZE
            
            # Secondary: RSI extreme alone (guarantees trades)
            if rsi_extreme_overbought and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
            
            # Funding contrarian: extreme long funding = short opportunity
            if funding_extreme_long and trend_4h_bearish and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL/TRANSITION: Use 1h SMA50 as tiebreaker ===
        if desired_signal == 0:
            if above_sma50 and rsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            if below_sma50 and rsi_extreme_overbought:
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
                # Hold long if 4h trend still bullish and RSI not overbought
                if trend_4h_bullish and rsi_1h[i] < 65:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h trend still bearish and RSI not oversold
                if trend_4h_bearish and rsi_1h[i] > 35:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h trend reverses
            if trend_4h_bearish and rsi_1h[i] > 55:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h trend reverses
            if trend_4h_bullish and rsi_1h[i] < 45:
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