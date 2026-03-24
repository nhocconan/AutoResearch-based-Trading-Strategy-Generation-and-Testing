#!/usr/bin/env python3
"""
Experiment #989: 15m Primary + 1h/1d HTF — Daily Pivot CPR + Camarilla Mean Reversion

Hypothesis: 15m entries at Daily Pivot/CPR levels with 1h trend filter and session timing
will capture intraday swings with high win rate while maintaining low trade frequency (40-100/yr).

Key innovations:
1. Daily CPR (Central Pivot Range): BC/TC from 1d HTF defines value area
   - Narrow CPR (<0.5% of price) = expansion day likely
   - Price above TC = bullish bias, below BC = bearish bias
2. Camarilla Levels (R3/R4, S3/S4): Mean reversion at extremes
   - Fade at R3/S3 in ranging market
   - Breakout at R4/S4 in trending market
3. 1h HMA(21) for intermediate trend direction
4. 15m RSI(7) for entry timing (oversold/overbought extremes)
5. Session filter: 00-12 UTC (London+NY overlap) for higher probability
6. Volume confirmation: taker_buy_volume > 1.5x 20-bar average

Why this should work on 15m:
- Daily pivots are institutional levels that matter across all timeframes
- Camarilla levels catch intraday extremes with high reversal probability
- 1h filter prevents counter-trend trades on lower TF noise
- Session filter avoids low-liquidity Asian session whipsaws
- Strict confluence (3+ factors) keeps trade count low (target 50-80/yr)

Entry conditions (balanced for trades):
- LONG = 1h bull + (price>TC OR narrow CPR) + RSI(7)<25 + session 00-12 + volume spike
- SHORT = 1h bear + (price<BC OR narrow CPR) + RSI(7)>75 + session 00-12 + volume spike
- Camarilla fade: RSI(7)>80 at R3 + 1h not strongly bull → short
- Camarilla fade: RSI(7)<20 at S3 + 1h not strongly bear → long

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for 15m frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_pivot_camarilla_rsi_1h1d_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_daily_pivots(df_1d):
    """
    Calculate Daily Pivot Points and CPR (Central Pivot Range)
    Uses previous day's OHLC to calculate today's levels
    
    Returns: dict with pivot, r1, r2, r3, s1, s2, s3, tc, bc, pivot_range
    """
    n = len(df_1d)
    pivots = {
        'pivot': np.full(n, np.nan),
        'r1': np.full(n, np.nan),
        'r2': np.full(n, np.nan),
        'r3': np.full(n, np.nan),
        's1': np.full(n, np.nan),
        's2': np.full(n, np.nan),
        's3': np.full(n, np.nan),
        'tc': np.full(n, np.nan),  # Top Central (CPR)
        'bc': np.full(n, np.nan),  # Bottom Central (CPR)
        'pivot_range': np.full(n, np.nan),  # TC - BC
    }
    
    for i in range(1, n):
        prev_high = df_1d['high'].iloc[i-1]
        prev_low = df_1d['low'].iloc[i-1]
        prev_close = df_1d['close'].iloc[i-1]
        
        pivot = (prev_high + prev_low + prev_close) / 3.0
        pivots['pivot'][i] = pivot
        
        # Standard pivot levels
        pivots['r1'][i] = 2.0 * pivot - prev_low
        pivots['s1'][i] = 2.0 * pivot - prev_high
        pivots['r2'][i] = pivot + (prev_high - prev_low)
        pivots['s2'][i] = pivot - (prev_high - prev_low)
        pivots['r3'][i] = prev_high + 2.0 * (pivot - prev_low)
        pivots['s3'][i] = prev_low - 2.0 * (prev_high - pivot)
        
        # CPR (Central Pivot Range)
        pivots['tc'][i] = (prev_high + prev_low) / 2.0
        pivots['bc'][i] = pivot
        pivots['pivot_range'][i] = abs(pivots['tc'][i] - pivots['bc'][i])
    
    return pivots

def calculate_camarilla_levels(close, high, low, prev_close):
    """
    Camarilla Pivot Levels
    R4/S4 = breakout levels
    R3/S3 = mean reversion levels
    """
    range_val = high - low
    
    r4 = prev_close + 1.5 * range_val
    r3 = prev_close + 1.2 * range_val
    r2 = prev_close + 1.1 * range_val
    r1 = prev_close + 1.05 * range_val
    
    s4 = prev_close - 1.5 * range_val
    s3 = prev_close - 1.2 * range_val
    s2 = prev_close - 1.1 * range_val
    s1 = prev_close - 1.05 * range_val
    
    return r4, r3, s3, s4

def calculate_session_hour(prices):
    """Extract UTC hour from open_time"""
    # open_time is in milliseconds since epoch
    timestamps = prices['open_time'].values / 1000.0
    hours = np.array([(int(ts) // 3600) % 24 for ts in timestamps])
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Daily pivots from 1d data
    daily_pivots = calculate_daily_pivots(df_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivots['pivot'])
    tc_aligned = align_htf_to_ltf(prices, df_1d, daily_pivots['tc'])
    bc_aligned = align_htf_to_ltf(prices, df_1d, daily_pivots['bc'])
    pivot_range_aligned = align_htf_to_ltf(prices, df_1d, daily_pivots['pivot_range'])
    
    # Camarilla levels need special handling - align the components
    prev_close_1d = np.roll(df_1d['close'].values, 1)
    prev_close_1d[0] = df_1d['close'].values[0]
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    camarilla_r3 = np.full(len(df_1d), np.nan)
    camarilla_s3 = np.full(len(df_1d), np.nan)
    camarilla_r4 = np.full(len(df_1d), np.nan)
    camarilla_s4 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        r4, r3, s3, s4 = calculate_camarilla_levels(
            df_1d['close'].values[i],
            high_1d[i-1],
            low_1d[i-1],
            prev_close_1d[i]
        )
        camarilla_r3[i] = r3
        camarilla_s3[i] = s3
        camarilla_r4[i] = r4
        camarilla_s4[i] = s4
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume average for spike detection
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    taker_ratio = taker_buy_vol / (volume + 1e-10)
    
    # Session hours
    session_hours = calculate_session_hour(prices)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h_aligned[i]) or np.isnan(pivot_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND (1h HMA) ===
        htf_1h_bull = close[i] > hma_1h_aligned[i]
        htf_1h_bear = close[i] < hma_1h_aligned[i]
        
        # Strong trend: price significantly above/below HMA
        hma_dist = (close[i] - hma_1h_aligned[i]) / (hma_1h_aligned[i] + 1e-10)
        htf_1h_strong_bull = hma_dist > 0.01  # >1% above HMA
        htf_1h_strong_bear = hma_dist < -0.01  # >1% below HMA
        
        # === DAILY PIVOT CONTEXT ===
        price_vs_pivot = (close[i] - pivot_aligned[i]) / (pivot_aligned[i] + 1e-10)
        above_pivot = price_vs_pivot > 0.0
        below_pivot = price_vs_pivot < 0.0
        
        # CPR context
        above_tc = close[i] > tc_aligned[i]
        below_bc = close[i] < bc_aligned[i]
        
        # Narrow CPR = expansion day likely (<0.5% of price)
        cpr_ratio = pivot_range_aligned[i] / (pivot_aligned[i] + 1e-10)
        narrow_cpr = cpr_ratio < 0.005
        
        # === CAMARILLA LEVELS ===
        at_r3 = abs(close[i] - r3_aligned[i]) / close[i] < 0.002  # Within 0.2%
        at_s3 = abs(close[i] - s3_aligned[i]) / close[i] < 0.002
        at_r4 = close[i] > r4_aligned[i]
        at_s4 = close[i] < s4_aligned[i]
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_7[i] < 25
        rsi_overbought = rsi_7[i] > 75
        rsi_extreme_oversold = rsi_7[i] < 20
        rsi_extreme_overbought = rsi_7[i] > 80
        
        # === SESSION FILTER (00-12 UTC = London+NY overlap) ===
        in_session = 0 <= session_hours[i] <= 12
        
        # === VOLUME SPIKE ===
        vol_spike = volume[i] > 1.5 * vol_avg_20[i] if not np.isnan(vol_avg_20[i]) else False
        taker_buying = taker_ratio[i] > 0.55
        taker_selling = taker_ratio[i] < 0.45
        
        # === ENTRY LOGIC (3+ CONFLUENCE REQUIRED) ===
        desired_signal = 0.0
        confluence_count = 0
        
        # LONG entries
        long_conditions = []
        
        # Condition 1: 1h trend bullish
        if htf_1h_bull:
            long_conditions.append(True)
        
        # Condition 2: Price above pivot or narrow CPR
        if above_pivot or narrow_cpr:
            long_conditions.append(True)
        
        # Condition 3: RSI oversold
        if rsi_oversold:
            long_conditions.append(True)
        
        # Condition 4: In session
        if in_session:
            long_conditions.append(True)
        
        # Condition 5: Volume confirmation
        if vol_spike or taker_buying:
            long_conditions.append(True)
        
        # Camarilla S3 mean reversion (works even if 1h not strongly bull)
        if at_s3 and rsi_extreme_oversold:
            long_conditions.append(True)
            long_conditions.append(True)  # Double weight for Camarilla
        
        confluence_count = sum(long_conditions)
        
        if confluence_count >= 3 and rsi_oversold:
            desired_signal = SIZE_STRONG if confluence_count >= 4 else SIZE_BASE
        
        # SHORT entries
        short_conditions = []
        
        # Condition 1: 1h trend bearish
        if htf_1h_bear:
            short_conditions.append(True)
        
        # Condition 2: Price below pivot or narrow CPR
        if below_pivot or narrow_cpr:
            short_conditions.append(True)
        
        # Condition 3: RSI overbought
        if rsi_overbought:
            short_conditions.append(True)
        
        # Condition 4: In session
        if in_session:
            short_conditions.append(True)
        
        # Condition 5: Volume confirmation
        if vol_spike or taker_selling:
            short_conditions.append(True)
        
        # Camarilla R3 mean reversion (works even if 1h not strongly bear)
        if at_r3 and rsi_extreme_overbought:
            short_conditions.append(True)
            short_conditions.append(True)  # Double weight for Camarilla
        
        confluence_count_short = sum(short_conditions)
        
        if confluence_count_short >= 3 and rsi_overbought:
            desired_signal = -SIZE_STRONG if confluence_count_short >= 4 else -SIZE_BASE
        
        # === BREAKOUT ENTRIES (at R4/S4) ===
        if at_r4 and htf_1h_strong_bull and rsi_7[i] > 50 and in_session:
            desired_signal = SIZE_STRONG
        
        if at_s4 and htf_1h_strong_bear and rsi_7[i] < 50 and in_session:
            desired_signal = -SIZE_STRONG
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals