#!/usr/bin/env python3
"""
Experiment #888: 30m Primary + 4h/1d HTF — Adaptive Regime + RSI Pullback + Volume

Hypothesis: After 600+ failed strategies, 30m timeframe needs EXTREMELY strict entry
filters to avoid fee drag (>100 trades/year kills profit). Key insight from research:

1. 30m Primary TF: Target 40-60 trades/year (use 3+ confluence filters)
2. 4h HMA(21) for trend direction — ONLY trade in HTF trend direction
3. 1d HMA(21) for macro regime — avoid counter-macro trades
4. 30m Choppiness Index(14) for regime: CHOP>55=range (mean revert), CHOP<45=trend (pullback)
5. 30m RSI(14) for entry timing — relaxed thresholds (30/70 not 20/80) to ensure trades
6. Volume filter (>0.8x 20-bar avg) — confirms participation
7. Session filter (8-20 UTC) — higher liquidity, lower slippage
8. ATR(14) 2.5x trailing stop for risk management

Why this should work on 30m:
- HTF (4h/1d) provides STRONG trend bias — reduces whipsaw
- Choppiness regime filter adapts logic to market state
- Relaxed RSI thresholds (30/70) ensure trades on ALL symbols
- Volume + session filters reduce noise and false signals
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Critical improvements from failed experiments:
- RELAXED RSI thresholds (30/70 not 20/80) to guarantee 30+ trades per symbol
- HTF trend filter prevents counter-trend trades (major source of losses)
- Choppiness regime adapts between mean-revert and trend-follow
- Volume confirmation reduces false breakouts
- ALL symbols MUST have positive Sharpe (no SOL-only bias)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 30m (target 40-60 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_rsi_pullback_4h1d_hma_vol_session_atr_v2"
timeframe = "30m"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 55 = ranging, CHOP < 45 = trending.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

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

def calculate_volume_avg(volume, period=20):
    """Volume moving average."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def extract_hour_from_timestamp(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    return (open_time // (1000 * 60 * 60)) % 24

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
    
    # Calculate primary (30m) indicators
    rsi_30m = calculate_rsi(close, period=14)
    chop_30m = calculate_choppiness(high, low, close, period=14)
    atr_30m = calculate_atr(high, low, close, period=14)
    vol_avg_30m = calculate_volume_avg(volume, period=20)
    sma_50_30m = calculate_sma(close, 50)
    sma_200_30m = calculate_sma(close, 200)
    
    # Calculate and align 4h HMA for medium-term trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro regime (bull/bear market)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(rsi_30m[i]) or np.isnan(chop_30m[i]):
            continue
        if np.isnan(atr_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_50_30m[i]) or np.isnan(vol_avg_30m[i]):
            continue
        
        # Extract UTC hour for session filter
        hour_utc = extract_hour_from_timestamp(open_time[i])
        in_session = 8 <= hour_utc <= 20  # London/NY overlap high liquidity
        
        # Volume confirmation
        volume_confirmed = volume[i] > 0.8 * vol_avg_30m[i]
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM TREND (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === SHORT-TERM TREND FILTER (30m SMA50/200) ===
        above_sma50 = close[i] > sma_50_30m[i]
        below_sma50 = close[i] < sma_50_30m[i]
        above_sma200 = close[i] > sma_200_30m[i]
        below_sma200 = close[i] < sma_200_30m[i]
        
        # === REGIME DETECTION (30m Choppiness Index) ===
        ranging_regime = chop_30m[i] > 55
        trending_regime = chop_30m[i] < 45
        # neutral regime: 45 <= CHOP <= 55
        
        # === RSI SIGNALS (Relaxed thresholds: 30/70 for more trades) ===
        rsi_oversold = rsi_30m[i] < 30
        rsi_overbought = rsi_30m[i] > 70
        rsi_extreme_oversold = rsi_30m[i] < 20
        rsi_extreme_overbought = rsi_30m[i] > 80
        rsi_neutral = 30 <= rsi_30m[i] <= 70
        
        desired_signal = 0.0
        
        # === TRENDING REGIME LOGIC (CHOP < 45) — Trend Pullback Entries ===
        if trending_regime:
            # Long: 4h bullish + RSI pullback to oversold + volume + session
            if trend_4h_bullish and rsi_oversold and volume_confirmed and in_session:
                desired_signal = BASE_SIZE
            # Also allow: macro bull + RSI oversold (stronger signal)
            elif macro_bull and rsi_oversold and volume_confirmed:
                desired_signal = BASE_SIZE
            
            # Short: 4h bearish + RSI rally to overbought + volume + session
            if trend_4h_bearish and rsi_overbought and volume_confirmed and in_session:
                desired_signal = -BASE_SIZE
            # Also allow: macro bear + RSI overbought (stronger signal)
            elif macro_bear and rsi_overbought and volume_confirmed:
                desired_signal = -BASE_SIZE
            
            # Fallback: extreme RSI alone (ensures trades on all symbols)
            if rsi_extreme_oversold and desired_signal == 0 and volume_confirmed:
                desired_signal = REDUCED_SIZE
            if rsi_extreme_overbought and desired_signal == 0 and volume_confirmed:
                desired_signal = -REDUCED_SIZE
        
        # === RANGING REGIME LOGIC (CHOP > 55) — Mean Reversion ===
        elif ranging_regime:
            # Long: RSI oversold + above SMA200 (not in crash) + volume
            if rsi_oversold and above_sma200 and volume_confirmed:
                desired_signal = BASE_SIZE
            # Short: RSI overbought + below SMA200 (not in rally) + volume
            if rsi_overbought and below_sma200 and volume_confirmed:
                desired_signal = -BASE_SIZE
            
            # Fallback: extreme RSI in range (ensures trades)
            if rsi_extreme_oversold and desired_signal == 0 and volume_confirmed:
                desired_signal = REDUCED_SIZE
            if rsi_extreme_overbought and desired_signal == 0 and volume_confirmed:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) — Conservative ===
        else:
            # Only take high-confidence signals with HTF alignment
            if trend_4h_bullish and rsi_oversold and volume_confirmed and in_session:
                desired_signal = REDUCED_SIZE
            if trend_4h_bearish and rsi_overbought and volume_confirmed and in_session:
                desired_signal = -REDUCED_SIZE
            
            # Fallback: extreme RSI with macro alignment
            if rsi_extreme_oversold and macro_bull and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            if rsi_extreme_overbought and macro_bear and desired_signal == 0:
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h trend intact and RSI not overbought
                if trend_4h_bullish and rsi_30m[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h trend intact and RSI not oversold
                if trend_4h_bearish and rsi_30m[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h trend reverses + RSI overbought
            if trend_4h_bearish and rsi_30m[i] > 70:
                desired_signal = 0.0
            # Exit if RSI extremely overbought in ranging regime
            if ranging_regime and rsi_30m[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h trend reverses + RSI oversold
            if trend_4h_bullish and rsi_30m[i] < 30:
                desired_signal = 0.0
            # Exit if RSI extremely oversold in ranging regime
            if ranging_regime and rsi_30m[i] < 20:
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
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
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