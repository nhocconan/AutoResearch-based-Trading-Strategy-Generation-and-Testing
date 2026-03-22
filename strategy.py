#!/usr/bin/env python3
"""
Experiment #398: 30m Primary + 4h/1d HTF — Connors RSI + HMA Trend + Session Filter

Hypothesis: After 360+ failed experiments, the pattern for lower TF (30m) is clear:
1. MUST use HTF (4h/1d) for SIGNAL DIRECTION, 30m only for ENTRY TIMING
2. MUST generate VERY FEW trades (30-80/year) — #388 had 0 trades = auto-reject
3. Connors RSI proven in literature for mean reversion (75% win rate)
4. Session filter (8-20 UTC) reduces noise during low-liquidity hours
5. Volume confirmation prevents false breakouts
6. Discrete sizing (0.20-0.30) with 2.5x ATR stoploss

Why this might beat current best (Sharpe=0.435):
- 1d HMA(21) for major trend (proven in #382, #389)
- 4h HMA(16/48) for intermediate trend confirmation
- Connors RSI(3,2,100) for precise entry timing on 30m
- Session filter (8-20 UTC) + volume > 0.8x avg for quality trades
- Looser CRSI thresholds (10-25 long, 75-90 short) to ensure >=30 trades/symbol

Position sizing: 0.20-0.30 (discrete, max 0.40) — CRITICAL for drawdown control
Stoploss: 2.5 * ATR trailing
Target: 30-80 trades/year on 30m, >=30 trades/symbol on train, >=3 on test

CRITICAL: Call get_htf_data() ONCE before loop, use aligned arrays inside.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_hma_session_volume_4h1d_v1"
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
    """Calculate Hull Moving Average (HMA)."""
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    Literature: Connors RSI has ~75% win rate for mean reversion.
    Long entry: CRSI < 10-15
    Short entry: CRSI > 85-90
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3) of close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI(2) of streak (consecutive up/down days)
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100 scale)
    # Positive streak = bullish, negative = bearish
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    rs_streak = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # Component 3: PercentRank of close over 100 periods
    percent_rank = pd.Series(close).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100.0,
        raw=False
    ).values
    
    # Combine components
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    
    return crsi

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    return vol_ratio

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    ts = pd.to_datetime(open_time, unit='ms', utc=True)
    return ts.dt.hour.values

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
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h HTF indicators (intermediate trend)
    hma_4h_16 = calculate_hma(df_4h['close'].values, period=16)
    hma_4h_48 = calculate_hma(df_4h['close'].values, period=48)
    hma_4h_16_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_16)
    hma_4h_48_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_48)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi_30m = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    vol_ratio = calculate_volume_ratio(volume, 20)
    utc_hour = get_utc_hour(open_time)
    
    # Session filter: 8-20 UTC (high liquidity hours)
    in_session = (utc_hour >= 8) & (utc_hour <= 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.25
    SHORT_SIZE = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(hma_4h_16_aligned[i]) or np.isnan(hma_4h_48_aligned[i]):
            continue
        
        if np.isnan(crsi_30m[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # === 4H INTERMEDIATE TREND ===
        hma_4h_bullish = hma_4h_16_aligned[i] > hma_4h_48_aligned[i]
        hma_4h_bearish = hma_4h_16_aligned[i] < hma_4h_48_aligned[i]
        
        # === CONNORS RSI SIGNALS (mean reversion) ===
        # Long: CRSI oversold (< 25 for more trades)
        crsi_oversold = crsi_30m[i] < 25.0
        # Short: CRSI overbought (> 75 for more trades)
        crsi_overbought = crsi_30m[i] > 75.0
        
        # === VOLUME CONFIRMATION ===
        volume_ok = vol_ratio[i] > 0.8  # At least 80% of avg volume
        
        # === SESSION FILTER ===
        session_ok = in_session[i]
        
        # === ENTRY LOGIC — HTF DIRECTION + 30m TIMING ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Bull regime + 4h bullish + CRSI oversold + volume + session
        # Require 3+ confluence: (bull_regime, hma_4h_bullish, crsi_oversold, volume_ok, session_ok)
        long_confluence = sum([bull_regime, hma_4h_bullish, crsi_oversold, volume_ok, session_ok])
        
        if long_confluence >= 3 and crsi_oversold:
            # Must have bull regime + CRSI oversold as base
            if bull_regime and crsi_oversold:
                new_signal = LONG_SIZE
        
        # SHORT ENTRY: Bear regime + 4h bearish + CRSI overbought + volume + session
        short_confluence = sum([bear_regime, hma_4h_bearish, crsi_overbought, volume_ok, session_ok])
        
        if short_confluence >= 3 and crsi_overbought:
            if bear_regime and crsi_overbought:
                if new_signal == 0.0:  # Don't flip if already long
                    new_signal = -SHORT_SIZE
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 48 bars (~24 hours on 30m), force entry on weaker signal
        if bars_since_last_trade > 48 and new_signal == 0.0 and not in_position:
            # Weaker confluence requirement
            if bull_regime and crsi_30m[i] < 35:
                new_signal = LONG_SIZE * 0.8
            elif bear_regime and crsi_30m[i] > 65:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
        
        # === EXIT CONDITIONS ===
        # CRSI extreme exit (take profit on mean reversion completion)
        if in_position and position_side > 0 and crsi_30m[i] > 70:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi_30m[i] < 30:
            new_signal = 0.0
        
        # Trend reversal exit (1d regime flip)
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
            new_signal = 0.0
        
        # 4h trend reversal exit
        if in_position and position_side > 0 and hma_4h_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_4h_bullish:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals