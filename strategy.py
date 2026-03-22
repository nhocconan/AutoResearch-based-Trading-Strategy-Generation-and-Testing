#!/usr/bin/env python3
"""
Experiment #528: 30m Primary + 4h/1d HTF — Connors RSI + HTF Trend + Session Filter

Hypothesis: After 473 failed strategies (mostly complex volspike/regime combos),
try a PROVEN mean-reversion approach with strict trade frequency control.

Key insights from failures:
- Volatility spike strategies: ALL failed (Sharpe negative)
- Choppiness index: Failed repeatedly
- Complex multi-filter: Too many conditions = 0 trades or whipsaw
- Lower TF (30m): MUST limit trades to 30-80/year or fee drag kills profit

This strategy uses:
1. 1d HMA(21) for MAJOR trend direction — only trade with HTF trend
2. 4h RSI(14) for momentum confirmation — avoid counter-momentum entries
3. 30m Connors RSI for entry timing — proven 75% win rate mean reversion
4. Session filter (8-20 UTC) — avoid low-liquidity Asian session whipsaw
5. Volume filter (>0.8x 20-bar avg) — confirm participation
6. ATR(14) 2.5x trailing stop for risk management

Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Long when CRSI < 10 (oversold) + HTF bull trend
- Short when CRSI > 90 (overbought) + HTF bear trend

Why this might work:
- Connors RSI is proven mean-reversion indicator (Larry Connors research)
- 1d trend filter prevents counter-trend trades (major failure mode)
- Session filter reduces low-liquidity whipsaw (critical for 30m)
- Simple confluence = consistent signals across BTC/ETH/SOL
- 30m TF with strict filters targets 40-60 trades/year (optimal fee/trade ratio)

Position sizing: 0.25 (smaller for lower TF, max 0.40 per rules)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_connors_hma1d_session_4h_v1"
timeframe = "30m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100):
    """
    Calculate Connors RSI (CRSI) - proven mean reversion indicator.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Reference: Larry Connors & Cesar Alvarez, "Short Term Trading Strategies That Work"
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, period_rsi)
    
    # Component 2: RSI of streak duration
    # Streak = consecutive up/down days
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of absolute streak values (period=2)
    streak_abs = np.abs(streak)
    streak_s = pd.Series(streak_abs)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    avg_streak_gain = streak_gain.ewm(span=period_streak, min_periods=period_streak, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=period_streak, min_periods=period_streak, adjust=False).mean()
    rs_streak = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    
    # Component 3: PercentRank of price change over 100 periods
    pct_change = close_s.pct_change(periods=1) * 100
    percent_rank = pd.Series(index=close_s.index, dtype=float)
    for i in range(period_rank, n):
        window = pct_change.iloc[i-period_rank:i]
        rank = (window < pct_change.iloc[i]).sum() / period_rank * 100
        percent_rank.iloc[i] = rank
    
    # Combine components
    crsi = (rsi_3 + rsi_streak.values + percent_rank.values) / 3.0
    
    return crsi

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)."""
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 1d HTF HMA for major trend direction
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Calculate 4h RSI for momentum confirmation
    rsi_4h_14 = calculate_rsi(df_4h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    rsi_4h_14_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h_14)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100)
    
    # Volume average for filter
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, smaller for 30m)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(rsi_4h_14_aligned[i]):
            continue
        if np.isnan(crsi[i]):
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        vol_ok = volume[i] > 0.8 * vol_avg_20[i]
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # 1d HMA slope for trend strength
        hma_slope_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_slope_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === 4H MOMENTUM CONFIRMATION ===
        mom_bull = rsi_4h_14_aligned[i] > 45.0  # Not bearish momentum
        mom_bear = rsi_4h_14_aligned[i] < 55.0  # Not bullish momentum
        
        # === CONNORS RSI EXTREMES (mean reversion signals) ===
        crsi_oversold = crsi[i] < 15.0  # Extreme oversold for long
        crsi_overbought = crsi[i] > 85.0  # Extreme overbought for short
        
        # === ENTRY LOGIC — STRICT CONFLUENCE FOR LOW TRADE COUNT ===
        new_signal = 0.0
        
        # LONG: CRSI oversold + bull regime + session + volume + momentum
        if crsi_oversold and bull_regime and in_session and vol_ok and mom_bull:
            new_signal = POSITION_SIZE
        
        # SHORT: CRSI overbought + bear regime + session + volume + momentum
        if new_signal == 0.0:
            if crsi_overbought and bear_regime and in_session and vol_ok and mom_bear:
                new_signal = -POSITION_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS (regime flip or CRSI mean reversion complete) ===
        # Exit long on regime flip or CRSI recovered
        if in_position and position_side > 0:
            if bear_regime and hma_slope_bear:
                new_signal = 0.0
            elif crsi[i] > 60.0:  # CRSI recovered from oversold
                new_signal = 0.0
        
        # Exit short on regime flip or CRSI mean reverted
        if in_position and position_side < 0:
            if bull_regime and hma_slope_bull:
                new_signal = 0.0
            elif crsi[i] < 40.0:  # CRSI recovered from overbought
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals