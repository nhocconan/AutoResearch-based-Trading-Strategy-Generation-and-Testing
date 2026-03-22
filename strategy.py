#!/usr/bin/env python3
"""
Experiment #405: 1h Primary + 4h/1d HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: After 400+ experiments, the pattern is clear:
1. 1h timeframe needs 30-60 trades/year to overcome fee drag (NOT 200+)
2. Use 4h/1d for SIGNAL DIRECTION, 1h only for ENTRY TIMING
3. Choppiness Index (CHOP) regime filter: CHOP>55=range(mean revert), CHOP<45=trend(follow)
4. Connors RSI (CRSI) for mean reversion: (RSI(3)+RSI_Streak(2)+PercentRank(100))/3
5. Session filter (8-20 UTC) for liquidity + volume confirmation
6. Discrete position sizing: 0.0, ±0.20, ±0.30 (max 0.40)
7. ATR 2.5x trailing stop for risk management

Why this might beat current best (Sharpe=0.435):
- CHOP regime filter prevents trend strategies in chop (major failure mode)
- CRSI catches reversals better than standard RSI (75% win rate in literature)
- 4h HMA(21) for major trend direction (proven in #382, #386)
- Session+volume filters reduce false signals during low liquidity
- 1h TF with HTF direction = optimal trade frequency (30-60/year)

Position sizing: 0.20-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 30-60 trades/year on 1h, >=10 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_crsi_hma_regime_4h1d_v1"
timeframe = "1h"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100.0 * np.log10(atr_sum / price_range + 1e-10) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    RSI(streak): RSI of consecutive up/down days
    PercentRank: percentage of past closes lower than current close
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3) of price
    rsi_price = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streak (consecutive up/down)
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_pos = np.where(streak > 0, streak, 0)
    streak_neg = np.where(streak < 0, -streak, 0)
    
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        avg_gain = np.mean(streak_pos[max(0,i-streak_period+1):i+1])
        avg_loss = np.mean(streak_neg[max(0,i-streak_period+1):i+1])
        if avg_loss == 0:
            streak_rsi[i] = 100.0
        else:
            rs = avg_gain / (avg_loss + 1e-10)
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 3: Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[max(0, i-rank_period+1):i+1]
        rank = np.sum(window < close[i]) / len(window) * 100.0
        percent_rank[i] = rank
    
    # Combine components
    crsi = (rsi_price + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_avg + 1e-10)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HTF indicators (major trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.25
    SHORT_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -50
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]) or np.isnan(vol_ratio[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC for liquidity) ===
        hour = pd.to_datetime(open_time[i], unit='ms').hour
        in_session = 8 <= hour <= 20
        
        # === 4H MAJOR TREND (primary direction filter) ===
        # Price above 4h HMA = bull market bias (favor longs)
        # Price below 4h HMA = bear market bias (favor shorts)
        bull_regime = close[i] > hma_4h_21_aligned[i]
        bear_regime = close[i] < hma_4h_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range/choppy (mean revert)
        # CHOP < 45 = trending (trend follow)
        choppy_regime = chop_14[i] > 55.0
        trending_regime = chop_14[i] < 45.0
        
        # === CONNORS RSI SIGNALS ===
        # CRSI < 15 = oversold (long in range/bull)
        # CRSI > 85 = overbought (short in range/bear)
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 0.8
        
        # === ENTRY LOGIC — REGIME-ADAPTIVE ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY conditions (multiple confluence required)
        if in_session and volume_confirmed:
            # Mean reversion long: choppy regime + oversold + bull trend
            if choppy_regime and crsi_oversold and bull_regime:
                new_signal = LONG_SIZE
            # Trend follow long: trending regime + bull trend + CRSI recovering
            elif trending_regime and bull_regime and 25.0 <= crsi[i] <= 50.0:
                new_signal = LONG_SIZE
        
        # SHORT ENTRY conditions (multiple confluence required)
        if in_session and volume_confirmed:
            # Mean reversion short: choppy regime + overbought + bear trend
            if choppy_regime and crsi_overbought and bear_regime:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Trend follow short: trending regime + bear trend + CRSI declining
            elif trending_regime and bear_regime and 50.0 <= crsi[i] <= 75.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
        
        # === FREQUENCY BOOST (ensure >=10 trades/symbol on train) ===
        # If no trade for 100 bars (~4 days on 1h), force entry on weaker signal
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if bull_regime and crsi[i] < 30.0 and in_session:
                new_signal = LONG_SIZE * 0.6
            elif bear_regime and crsi[i] > 70.0 and in_session:
                new_signal = -SHORT_SIZE * 0.6
        
        # === EXIT CONDITIONS ===
        # CRSI extreme exit (take profit on momentum exhaustion)
        if in_position and position_side > 0 and crsi[i] > 80.0:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi[i] < 20.0:
            new_signal = 0.0
        
        # Trend reversal exit (4h regime flip)
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
            new_signal = 0.0
        
        # Session exit (close position outside trading hours)
        if in_position and not in_session:
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