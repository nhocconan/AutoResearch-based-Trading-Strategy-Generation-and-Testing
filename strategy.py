#!/usr/bin/env python3
"""
Experiment #038: 30m KAMA Pullback + 4h HMA Trend + ATR Risk Management

Hypothesis: 30m primary with 4h HTF trend filter will capture pullback entries
within established trends. Key design to avoid 30m failures:
1. 4h HMA(21) for major trend direction (call ONCE before loop via mtf_data)
2. 30m KAMA(14) for adaptive local trend (less whipsaw than EMA)
3. RSI(14) pullback entries: long when RSI<50 in uptrend, short when RSI>50 in downtrend
4. ATR(14) for stoploss (2.5x) - mandatory risk management
5. Minimum 20 bars between trades (~10 hours) to control frequency
6. Discrete sizing: 0.25 base, 0.30 strong trend alignment, 0.20 weak

Why this should work on 30m:
- 4h HTF filter ensures we only trade WITH the major trend (avoids 2022 whipsaw)
- KAMA adapts to volatility - slower in chop, faster in trends
- RSI pullback (not extreme) ensures trades trigger frequently enough
- 20-bar minimum between trades targets 40-80 trades/year (optimal for 30m)
- Simple logic avoids 0-trade failures seen in #028, #029, #030, #032, #035

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete (conservative for lower TF)
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_kama_pullback_4h_hma_atr_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_avg = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_avg = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    loss_avg = np.where(loss_avg == 0, 1e-10, loss_avg)
    rs = gain_avg / loss_avg
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_kama(close, period=14, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average.
    
    KAMA adapts to market volatility:
    - Fast in trending markets (low noise)
    - Slow in choppy markets (high noise)
    
    Efficiency Ratio (ER) = |net change| / sum of absolute changes
    SC (Smoothing Constant) = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Efficiency Ratio
    net_change = np.abs(close - np.roll(close, period))
    net_change[:period] = np.nan
    
    sum_changes = np.zeros(n)
    for i in range(period, n):
        sum_changes[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1)))
    
    er = net_change / np.where(sum_changes == 0, 1e-10, sum_changes)
    er = np.nan_to_num(er, nan=0)
    
    # Smoothing constants
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama[period] = close[period]
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_diff = high[i] - high[i-1]
        minus_diff = low[i-1] - low[i]
        
        if plus_diff > minus_diff and plus_diff > 0:
            plus_dm[i] = plus_diff
        if minus_diff > plus_diff and minus_diff > 0:
            minus_dm[i] = minus_diff
    
    # Smoothed DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI and ADX
    plus_di = 100 * plus_dm_s / np.where(tr_s == 0, 1e-10, tr_s)
    minus_di = 100 * minus_dm_s / np.where(tr_s == 0, 1e-10, tr_s)
    
    dx = 100 * np.abs(plus_di - minus_di) / np.where(plus_di + minus_di == 0, 1e-10, plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA trend
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_14 = calculate_kama(close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    WEAK_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(kama_14[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        
        # === 4H HTF TREND BIAS ===
        # Price above 4h HMA = bullish bias, below = bearish
        htf_bullish = close[i] > hma_4h_21_aligned[i]
        htf_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === 30M LOCAL TREND (KAMA) ===
        local_bullish = close[i] > kama_14[i]
        local_bearish = close[i] < kama_14[i]
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 20  # Lower threshold for 30m
        adx_weak = adx_14[i] <= 20
        
        # === RSI PULLBACK FILTER (loose to ensure trades trigger) ===
        # Long: RSI < 50 (pullback in uptrend)
        # Short: RSI > 50 (pullback in downtrend)
        rsi_pullback_long = rsi_14[i] < 50
        rsi_pullback_short = rsi_14[i] > 50
        
        # === POSITION SIZING BASED ON TREND STRENGTH ===
        if htf_bullish and local_bullish and adx_strong:
            current_size = STRONG_SIZE
        elif htf_bullish and local_bullish:
            current_size = BASE_SIZE
        elif htf_bullish:
            current_size = WEAK_SIZE
        elif htf_bearish and local_bearish and adx_strong:
            current_size = STRONG_SIZE
        elif htf_bearish and local_bearish:
            current_size = BASE_SIZE
        elif htf_bearish:
            current_size = WEAK_SIZE
        else:
            current_size = BASE_SIZE
        
        # === ENTRY LOGIC (PULLBACK WITHIN TREND) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: 4h bullish + RSI pullback < 50 + price > KAMA
        if htf_bullish and rsi_pullback_long and local_bullish:
            new_signal = current_size
        
        # SHORT ENTRY: 4h bearish + RSI pullback > 50 + price < KAMA
        elif htf_bearish and rsi_pullback_short and local_bearish:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 25 bars (~12.5 hours on 30m), allow weaker entry
        # This ensures we generate enough trades (target 40-80/year)
        if bars_since_last_trade > 25 and new_signal == 0.0 and not in_position:
            if htf_bullish and rsi_14[i] < 55:  # Looser RSI
                new_signal = current_size * 0.8
            elif htf_bearish and rsi_14[i] > 45:  # Looser RSI
                new_signal = -current_size * 0.8
        
        # === ENFORCE MINIMUM BARS BETWEEN TRADES ===
        # Don't flip position within 20 bars (~10 hours)
        if bars_since_last_trade < 20 and in_position:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 4h trend turns bearish
            if position_side > 0 and htf_bearish:
                trend_reversal = True
            # Exit short if 4h trend turns bullish
            if position_side < 0 and htf_bullish:
                trend_reversal = True
        
        # === RSI EXTREME EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Exit long when RSI becomes overbought
            if position_side > 0 and rsi_14[i] > 70:
                rsi_exit = True
            # Exit short when RSI becomes oversold
            if position_side < 0 and rsi_14[i] < 30:
                rsi_exit = True
        
        # Apply stoploss or reversals
        if stoploss_triggered or trend_reversal or rsi_exit:
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
                # Position flip - count as new trade
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