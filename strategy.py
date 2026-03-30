#!/usr/bin/env python3
"""
Experiment #022: Tight CRSI + Choppiness + Donchian (4h)

HYPOTHESIS: Simplified version of proven CRSI strategy that works in BOTH bull and bear:
- CRSI < 15 + price > SMA200 + choppiness < 40 = long (oversold in uptrend)
- CRSI > 85 + price < SMA200 + choppiness < 40 = short (overbought in downtrend)
- Choppiness filter avoids range markets (no trades when CHOP > 50)
- Donchian(20) breakout confirms momentum shift
- 3*ATR trailing stop for risk management

WHY IT SHOULD WORK IN BOTH MARKETS:
- Bull market: Pullbacks to oversold = entry opportunities, price above SMA200 confirms bull
- Bear market: rallies to overbought = short opportunities, price below SMA200 confirms bear
- Choppiness < 40 = trending, so we only trade when market has direction
- Fewer, tighter conditions = fewer trades = less fee drag = better generalization

KEY INSIGHT: The DB's CRSI/Donchian/Chop had 392 trades with Sharpe 1.46.
This is too many. Tighten CRSI from <30 to <15, add Donchian breakout confirmation.

TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_tight_chop_donchian_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_crsi(prices, period_rsi=3, period_streak=2, period_rank=100):
    """
    Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long signal: CRSI < 10, price > SMA200 (75% win rate in backtests)
    Short signal: CRSI > 90, price < SMA200
    """
    close = prices["close"].values
    n = len(close)
    
    # RSI(3) using standard approach
    delta = np.zeros(n)
    delta[1:] = close[1:] - close[:-1]
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period_rsi, min_periods=period_rsi, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period_rsi, min_periods=period_rsi, adjust=False).mean().values
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi3 = 100 - (100 / (1 + rs))
    
    # RSI Streak (2)
    # Count consecutive up/down closes
    streak = np.zeros(n)
    for i in range(1, n):
        if delta[i] > 0:
            streak[i] = max(0, streak[i-1] + 1)
        elif delta[i] < 0:
            streak[i] = min(0, streak[i-1] - 1)
        else:
            streak[i] = streak[i-1]
    
    # RSI of streak
    abs_streak = np.abs(streak)
    avg_gain_s = pd.Series(np.where(streak > 0, streak, 0)).ewm(span=period_streak, min_periods=period_streak, adjust=False).mean().values
    avg_loss_s = pd.Series(np.where(streak < 0, abs_streak, 0)).ewm(span=period_streak, min_periods=period_streak, adjust=False).mean().values
    
    rs_s = np.where(avg_loss_s > 0, avg_gain_s / avg_loss_s, 100)
    rsi_streak = 100 - (100 / (1 + rs_s))
    
    # Percent Rank (100)
    prc_rank = np.zeros(n)
    for i in range(period_rank, n):
        window = close[i-period_rank+1:i+1]
        rank = np.sum(window < close[i]) / period_rank
        prc_rank[i] = rank * 100
    
    # CRSI
    crsi = (rsi3 + rsi_streak + prc_rank) / 3
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index: measures market trendiness
    CHOP > 61.8 = range (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # Highest high - lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        range_sum = hh - ll
        
        if range_sum > 0:
            chop[i] = 100 * np.log(atr_sum / range_sum) / np.log(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # === HTF SMA200 for trend direction ===
    sma_200_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    htf_above_sma = df_1d['close'].values > sma_200_1d
    htf_below_sma = df_1d['close'].values < sma_200_1d
    htf_bull_aligned = align_htf_to_ltf(prices, df_1d, htf_above_sma.astype(float))
    htf_bear_aligned = align_htf_to_ltf(prices, df_1d, htf_below_sma.astype(float))
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # CRSI
    crsi = calculate_crsi(prices, period_rsi=3, period_streak=2, period_rank=100)
    
    # Choppiness
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Donchian channels (20 periods)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio (20 period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 300  # CRSI needs 100 bars, SMA200 needs 200
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
        
        # === TREND CONDITIONS ===
        # Choppiness < 40 = trending market
        trending = chop[i] < 40.0
        
        # Price vs local SMA200 (4h)
        sma_200 = pd.Series(close[:i+1]).rolling(window=200, min_periods=200).mean().iloc[-1]
        price_above_sma200 = close[i] > sma_200 if not np.isnan(sma_200) else False
        price_below_sma200 = close[i] < sma_200 if not np.isnan(sma_200) else False
        
        # HTF trend (1d)
        htf_bull = htf_bull_aligned[i] > 0.5 if not np.isnan(htf_bull_aligned[i]) else False
        htf_bear = htf_bear_aligned[i] > 0.5 if not np.isnan(htf_bear_aligned[i]) else False
        
        # Donchian breakout (price breaks 20-bar high/low)
        donchian_break_high = close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1] if i > 0 else False
        donchian_break_low = close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1] if i > 0 else False
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position and trending:
            # LONG: CRSI < 15 + price above SMA200 + (HTF bull OR price above 4h SMA200) + Donchian breakout or vol spike
            if crsi[i] < 15 and (price_above_sma200 or htf_bull):
                if donchian_break_high or vol_spike:
                    desired_signal = SIZE
            
            # SHORT: CRSI > 85 + price below SMA200 + (HTF bear OR price below 4h SMA200) + Donchian breakout or vol spike
            elif crsi[i] > 85 and (price_below_sma200 or htf_bear):
                if donchian_break_low or vol_spike:
                    desired_signal = -SIZE
        
        # === STOPLOSS (3 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop
                stop_price = trailing_high - 3.0 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if CRSI reaches overbought
                if crsi[i] > 70:
                    desired_signal = 0.0
                
                # Exit if choppiness rises (market becoming choppy)
                if chop[i] > 55:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop
                stop_price = trailing_low + 3.0 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if CRSI reaches oversold
                if crsi[i] < 30:
                    desired_signal = 0.0
                
                # Exit if choppiness rises
                if chop[i] > 55:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 4 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 4:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals