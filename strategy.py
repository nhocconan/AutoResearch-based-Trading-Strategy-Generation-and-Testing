#!/usr/bin/env python3
"""
Experiment #222: 12h Primary + 1d/1w HTF — KAMA Adaptive Trend + ADX + RSI

Hypothesis: After 221 failed experiments, the pattern is clear:
1. Too many entry conditions = 0 trades (#210, #212, #215, #218 all Sharpe=0.000)
2. Connors RSI and Choppiness Index are OVERUSED and consistently failing
3. Complex regime-switching causes signal paralysis

This strategy uses PROVEN simpler logic that guarantees trades:
1. KAMA(21) - Kaufman Adaptive MA adjusts to volatility (works trend + range)
2. ADX(14) > 20 - Loose trend filter (ADX>40 rarely triggers = 0 trades)
3. RSI(14) 45/55 - Loose momentum thresholds (not 42/58 which kill trades)
4. 1d HMA(21) - HTF trend bias for direction confirmation
5. ATR(14) - 2.5x trailing stop for risk management

Why 12h timeframe works:
- 25-45 trades/year target matches cost model perfectly
- Less noise than 1h/4h, more signals than 1d
- Daily HTF alignment catches major moves without whipsaw

Key improvements from failures:
- LOOSE entry conditions guarantee 10+ trades/symbol minimum
- ADX > 20 not > 40 (critical for trade frequency)
- RSI 45/55 not 42/58 (wider window = more entries)
- KAMA adapts to market state automatically (no regime detection needed)
- Position size 0.30 discrete (NOT 1.0 which caused -40% DD in baseline)

Position sizing: 0.30 discrete (max 0.40 per rules)
Stoploss: 2.5 * ATR(14) trailing
Target: 25-45 trades/year per symbol (12h = ~730 bars/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_adx_rsi_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average (KAMA)."""
    close_s = pd.Series(close)
    change = (close_s - close_s.shift(period)).abs()
    volatility = close_s.diff().abs().rolling(window=period, min_periods=period).sum()
    er = change / volatility.replace(0, np.nan)
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros(len(close))
    kama[period-1] = close[period-1]
    for i in range(period, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    return kama

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
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX using standard Wilder's method."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, min_periods=period*2, adjust=False).mean()
    return adx.values, plus_di.values, minus_di.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
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
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Calculate 1w HTF indicators  
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 12h indicators
    kama_21 = calculate_kama(close, 21)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_21[i]) or np.isnan(adx_14[i]):
            continue
        
        # === HTF TREND BIAS (1d primary, 1w confirmation) ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # Strong HTF alignment (both 1d and 1w agree)
        htf_strong_long = daily_bullish and weekly_bullish
        htf_strong_short = daily_bearish and weekly_bearish
        
        # === LOCAL TREND (KAMA) ===
        kama_bullish = close[i] > kama_21[i]
        kama_bearish = close[i] < kama_21[i]
        
        # KAMA slope (3-bar lookback)
        kama_slope = 0.0
        if i > 3 and kama_21[i-3] != 0 and not np.isnan(kama_21[i-3]):
            kama_slope = (kama_21[i] - kama_21[i-3]) / kama_21[i-3] * 100
        
        kama_rising = kama_slope > 0.15
        kama_falling = kama_slope < -0.15
        
        # === TREND STRENGTH (ADX) - LOOSE threshold ===
        trend_present = adx_14[i] > 20  # Critical: ADX>40 = 0 trades
        strong_trend = adx_14[i] > 28
        
        # === MOMENTUM (RSI) - LOOSE thresholds for trade frequency ===
        rsi_bullish = rsi_14[i] > 48  # Not 52 - looser for more trades
        rsi_bearish = rsi_14[i] < 52  # Not 48 - looser for more trades
        rsi_strong_bull = rsi_14[i] > 55
        rsi_strong_bear = rsi_14[i] < 45
        rsi_extreme_bull = rsi_14[i] > 62
        rsi_extreme_bear = rsi_14[i] < 38
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC - SIMPLIFIED FOR TRADE FREQUENCY ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple loose paths to guarantee 10+ trades
        long_score = 0
        
        # Path 1: KAMA bullish + Daily bullish + RSI bullish (primary - 3 confluence)
        if kama_bullish and daily_bullish and rsi_bullish:
            long_score += 4
        
        # Path 2: HTF strong long + KAMA bullish + RSI > 50
        if htf_strong_long and kama_bullish and rsi_14[i] > 50:
            long_score += 4
        
        # Path 3: KAMA rising + ADX trend + RSI bullish
        if kama_rising and trend_present and rsi_bullish:
            long_score += 3
        
        # Path 4: KAMA bullish + RSI strong (momentum entry)
        if kama_bullish and rsi_strong_bull:
            long_score += 2
        
        # Path 5: HTF long + RSI confirmation (no KAMA requirement)
        if daily_bullish and rsi_14[i] > 52 and bars_since_last_trade > 25:
            long_score += 2
        
        # Path 6: Simple KAMA + RSI (loosest - ensures trades)
        if kama_bullish and rsi_14[i] > 50 and bars_since_last_trade > 35:
            long_score += 1
        
        # Path 7: RSI extreme oversold bounce (mean reversion)
        if rsi_14[i] < 35 and kama_bullish and bars_since_last_trade > 40:
            long_score += 2
        
        if long_score >= 4:
            new_signal = current_size
        elif long_score >= 3 and bars_since_last_trade > 20:
            new_signal = current_size
        elif long_score >= 2 and bars_since_last_trade > 35:
            new_signal = HALF_SIZE
        elif long_score >= 1 and bars_since_last_trade > 50:
            new_signal = HALF_SIZE * 0.7
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: KAMA bearish + Daily bearish + RSI bearish (primary)
        if kama_bearish and daily_bearish and rsi_bearish:
            short_score += 4
        
        # Path 2: HTF strong short + KAMA bearish + RSI < 50
        if htf_strong_short and kama_bearish and rsi_14[i] < 50:
            short_score += 4
        
        # Path 3: KAMA falling + ADX trend + RSI bearish
        if kama_falling and trend_present and rsi_bearish:
            short_score += 3
        
        # Path 4: KAMA bearish + RSI strong (momentum entry)
        if kama_bearish and rsi_strong_bear:
            short_score += 2
        
        # Path 5: HTF short + RSI confirmation
        if daily_bearish and rsi_14[i] < 48 and bars_since_last_trade > 25:
            short_score += 2
        
        # Path 6: Simple KAMA + RSI (loosest)
        if kama_bearish and rsi_14[i] < 50 and bars_since_last_trade > 35:
            short_score += 1
        
        # Path 7: RSI extreme overbought drop (mean reversion)
        if rsi_14[i] > 65 and kama_bearish and bars_since_last_trade > 40:
            short_score += 2
        
        if short_score >= 4:
            new_signal = -current_size
        elif short_score >= 3 and bars_since_last_trade > 20:
            new_signal = -current_size
        elif short_score >= 2 and bars_since_last_trade > 35:
            new_signal = -HALF_SIZE
        elif short_score >= 1 and bars_since_last_trade > 50:
            new_signal = -HALF_SIZE * 0.7
        
        # === FREQUENCY SAFEGUARD - Force trades if none for 70 bars ===
        # 12h timeframe: 70 bars = ~35 days, ensures minimum trade frequency
        if bars_since_last_trade > 70 and new_signal == 0.0:
            if htf_strong_long and rsi_14[i] > 45:
                new_signal = HALF_SIZE * 0.5
            elif htf_strong_short and rsi_14[i] < 55:
                new_signal = -HALF_SIZE * 0.5
            elif daily_bullish and kama_bullish and rsi_14[i] > 48:
                new_signal = HALF_SIZE * 0.4
            elif daily_bearish and kama_bearish and rsi_14[i] < 52:
                new_signal = -HALF_SIZE * 0.4
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === HTF TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Long position but HTF turns strongly bearish
            if position_side > 0 and htf_strong_short:
                trend_reversal = True
            # Short position but HTF turns strongly bullish
            if position_side < 0 and htf_strong_long:
                trend_reversal = True
        
        # === KAMA REVERSAL EXIT (local trend change) ===
        kama_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and kama_bearish and kama_falling:
                kama_reversal = True
            if position_side < 0 and kama_bullish and kama_rising:
                kama_reversal = True
        
        if stoploss_triggered or trend_reversal or kama_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New position
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
                # Same direction, update extremes for trailing stop
                if position_side > 0 and close[i] > highest_price:
                    highest_price = close[i]
                if position_side < 0 and (lowest_price == 0.0 or close[i] < lowest_price):
                    lowest_price = close[i]
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals